from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import classification_report

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_spacy.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid_spacy.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_spacy.txt")), "test")

    def get_labels(self):
        return ["O", 'B-ElectricityMeasuringTool', 'B-ForceUnit', 'I-VolumeUnit', 'B-PhaseTransitionPoint', 'I-GeologicalEonsErasPeriodsEpochsAges',
                'B-Cell', 'I-Toxins', 'B-CelestialObject', 'B-PartsOfTheFoodChain', 'I-ScientificAssociationsAdministrations', 'B-Currents', 'I-SolidMatter',
                'I-ConcludingResearch', 'I-MagneticDirectionMeasuringTool', 'I-ActUponSomething', 'I-InheritedBehavior', 'B-Pattern', 'I-MoneyTerms', 'I-ManmadeObjects',
                'B-GeologicTheories', 'I-OrganicCompounds', 'B-GroupsOfOrganisms', 'I-CellProcesses', 'B-Bird', 'B-ColorChangingActions', 'I-Circuits', 'B-Touch', 'I-ObservationPlacesEGObservatory', 'I-TypesOfEvent', 'B-Age', 'I-Continents', 'B-Ability', 'B-RelativeNumber', 'B-MassUnit', 'I-Relations', 'B-LightMovement', 'B-SimpleMachines', 'B-Method', 'B-WordsForData', 'B-NutritiveSubstancesForAnimalsOrPlants', 'I-StarLayers', 'B-OrganicCompounds', 'I-TypesOfWaterInBodiesOfWater', 'B-AirVehicle', 'B-StarTypes', 'I-OtherEnergyResources', 'B-BirdAnimalPart', 'B-Geography', 'B-EclipseEvents', 'B-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'I-Forests', 'B-ConstructiveDestructiveForces', 'I-SpecificNamedBodiesOfWater', 'I-ActionsForTides', 'I-MuscularSystem', 'I-Representation', 'B-ActionsForTides', 'B-HumanPart', 'I-Amphibian', 'I-TheoryOfMatter', 'I-Event', 'I-ConstructiveDestructiveForces', 'B-PerformAnActivity', 'I-Taxonomy', 'I-Pressure', 'B-Development', 'I-PERCENT', 'I-Comet', 'I-Sedimentary', 'I-Property', 'I-Bacteria', 'I-ExamplesOfSounds', 'B-GeometricSpatialObjects', 'B-EmergencyServices', 'I-VariablesControls', 'B-LiquidHoldingContainersRecepticles', 'I-MuscularSystemActions', 'I-FossilRecordTimeline', 'B-SpeedUnit', 'B-Quality', 'B-Event', 'B-SpacecraftHumanRated', 'B-IncreaseDecrease', 'B-GeopoliticalLocations', 'B-Metabolism', 'I-PhysicalProperty', 'I-PartsOfTheReproductiveSystem', 'I-States', 'I-NaturalSelection', 'B-NUMBER', 'B-Substances', 'I-PlantNutrients', 'I-VerbsForLocate', 'I-SpaceProbes', 'B-Gravity', 'B-OuterPlanets', 'B-DistanceMeasuringTools', 'B-Calculations', 'I-Homeostasis', 'I-Vehicle', 'B-Surpass', 'I-Examine', 'I-EarthPartsGrossGroundAtmosphere', 'I-ObservationTechniques', 'B-PropertyOfMotion', 'I-RelativeNumber', 'B-AnimalClassificationMethod', 'B-ResultsOfDecomposition', 'I-Distance', 'I-Viewpoint', 'B-Agriculture', 'I-Size', 'I-PhaseTransitionPoint', 'I-Asteroid', 'I-Agriculture', 'B-LiquidMovement', 'B-ElectricalProperty', 'I-NaturalMaterial', 'B-ORDINAL', 'B-ElectricAppliance', 'I-Insect', 'I-Star', 'B-StructuralAdaptation', 'I-PlantCellPart', 'I-Occur', 'B-WaterVehicle', 'I-PartsOfEarthLayers', 'I-AnalyzingResearch', 'B-EnvironmentalPhenomena', 'B-FossilFuel', 'B-Arachnid', 'B-NationalityOrigin', 'I-BehavioralAdaptation', 'B-Uptake', 'B-DistanceComparison', 'B-TransferEnergy', 'B-CosmologicalTheoriesBigBangBigCrunch', 'B-WrittenMedia', 'B-GeographicFormationParts', 'B-Rock', 'I-GroupsOfOrganisms', 'B-LivingThing', 'B-GeneticProcesses', 'B-Animal', 'I-PartsOfEndocrineSystem', 'I-Collect', 'I-PerformingResearch', 'B-Sky', 'B-Blood', 'B-ReplicatingResearch', 'I-PERSON', 'I-PartsOfTheCirculatorySystem', 'I-Blood', 'I-Behaviors', 'B-Width', 'I-AnimalCellPart', 'I-Nutrition', 'I-TemperatureMeasuringTools', 'B-Meteorology', 'I-WeatherPhenomena', 'I-LearnedBehavior', 'I-CosmologicalTheoriesBigBangBigCrunch', 'B-Temperature', 'B-GaseousMovement', 'B-GalaxyParts', 'B-OrganismRelationships', 'B-Force', 'I-Consumption', 'I-Senses', 'I-ResultsOfDecomposition', 'I-GeologicTheories', 'I-CastFossilMoldFossil', 'I-ImmuneSystem', 'B-Products', 'B-PartsOfARepresentation', 'I-PhasesOfWater', 'B-ThermalEnergy', 'I-TimesOfDayDayNight', 'I-Planet', 'B-MeasuringSpeed', 'I-GeographicFormationParts', 'B-IllnessPreventionCuring', 'I-LiquidMovement', 'I-UnderwaterEcosystem', 'B-SensoryTerms', 'I-Relevant', 'B-ElectricalUnit', 'I-PropertyOfProduction', 'B-Reptile', 'B-TypesOfIllness', 'B-Compete', 'B-SystemProcessStages', 'I-LivingDying', 'I-PowerUnit', 'B-ImportanceComparison', 'I-Release', 'B-Rarity', 'B-PartsOfEndocrineSystem', 'I-PropertyOfMotion', 'B-VehicularSystemsParts', 'B-DigestiveSubstances', 'I-ElectricalUnit', 'B-ElectricityAndCircuits', 'I-SystemAndFunctions', 'I-PhysicalActivity', 'I-VolumeMeasuringTool', 'B-Galaxy', 'I-CombineAdd', 'B-Gender', 'B-ChangesToResources', 'I-Stability', 'B-CelestialLightOnEarth', 'I-IntegumentarySystem', 'B-PartsOfTheSkeletalSystem', 'I-Genetics', 'B-Conductivity', 'I-Traffic', 'B-Release', 'I-DigestiveSubstances', 'I-ElementalComponents', 'I-Comparisons', 'B-GeneticRelations', 'I-NervousSystem', 'I-MineralFormations', 'I-VisualComparison', 'I-ScientificMeetings', 'B-LOCATION', 'I-AreaUnit', 'B-Magnetic', 'B-NaturalPhenomena', 'B-PlantPart', 'B-MuscularSystemActions', 'I-PartsOfTheNervousSystem', 'I-ClassesOfElements', 'B-FossilTypesIndexFossil', 'I-Countries', 'B-Negations', 'I-RepresentingElementsAndMolecules', 'B-IntegumentarySystem', 'B-Difficulty', 'B-ReptileAnimalPart', 'B-Length', 'I-Appliance', 'B-Separate', 'B-Examine', 'I-ChangeInLocation', 'I-MeasurementsForHeatChange', 'B-Satellite', 'I-OtherAnimalProperties', 'I-Move', 'B-Observe', 'B-Break', 'B-OtherGeographicWords', 'I-Products', 'B-ScientificTheoryExperimentationAndHistory', 'I-Frequency', 'B-Relevant', 'B-MineralFormations', 'B-PartsOfWaves', 'B-PartsOfAGroup', 'B-StateOfBeing', 'B-MagneticEnergy', 'B-Experimentation', 'B-Viewpoint', 'B-Monera', 'I-Believe', 'I-Exemplar', 'B-Employment', 'I-AbsorbEnergy', 'I-Speed', 'B-HeatingAppliance', 'B-SeedlessVascular', 'I-Gene', 'I-Mutation', 'I-ScientificTheoryExperimentationAndHistory', 'I-ElectricityAndCircuits', 'B-NaturalResources', 'B-ChangeInLocation', 'B-Composition', 'I-ResistanceStrength', 'B-Star', 'I-PartsOfObservationInstruments', 'B-AmountComparison', 'B-Mutation', 'I-ManMadeGeographicFormations', 'B-Shape', 'I-Uptake', 'B-ObjectQuantification', 'I-OrganismRelationships', 'I-LightMovement', 'B-Move', 'B-GroupsOfScientists', 'I-ElectricAppliance', 'B-Numbers', 'I-LivingThing', 'B-PH', 'B-Classify', 'I-ChemicalProperty', 'I-LOCATION', 'B-Problem', 'B-FormChangingActions', 'B-YearNumerals', 'B-DistanceUnit', 'B-TheoryOfMatter', 'I-SoundMeasuringTools', 'B-GuidelinesAndRules', 'B-CelestialMovement', 'B-AstronomyAeronautics', 'B-PartsOfTheCirculatorySystem', 'B-ConstructionTools', 'B-Injuries', 'B-Human', 'I-Create', 'I-SpeedUnit', 'I-Quality', 'I-ElectricityGeneration', 'B-AcademicMedia', 'B-WavePerception', 'B-Toxins', 'B-LandVehicle', 'I-Igneous', 'I-Substances', 'B-ApparentCelestialMovement', 'B-AnimalCellPart', 'I-Harm', 'B-TrueFormFossil', 'B-Exemplar', 'I-FossilFuel', 'B-Scientists', 'I-Animal', 'I-Scientists', 'B-PrenatalOrganismStates', 'I-FoodChain', 'I-EcosystemsEnvironment', 'B-Appliance', 'B-QuestionActivityType', 'B-PartsOfABuilding', 'B-Inertia', 'I-PartsOfTheExcretorySystem', 'I-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'I-TrueFormFossil', 'B-FeedbackMechanism', 'B-ClassesOfElements', 'B-EarthPartsGrossGroundAtmosphere', 'I-SpaceMissionsEGApolloGeminiMercury', 'B-GapsAndCracks', 'B-Igneous', 'I-TraceFossil', 'B-WeatherPhenomena', 'B-ReproductiveSystem', 'B-PostnatalOrganismStages', 'B-CoolingToolsFood', 'I-WaterVehiclePart', 'I-ObjectPart', 'I-ORGANIZATION', 'I-AtmosphericLayers', 'B-PartsOfTheEye', 'B-Locations', 'I-EnergyWaves', 'I-Unit', 'B-ElectricalEnergy', 'B-Relations', 'B-GeographicFormations', 'B-AngleMeasuringTools', 'I-Length', 'I-ActionsForAnimals', 'B-Energy', 'B-VolumeMeasuringTool', 'I-SimpleMachines', 'I-BirdAnimalPart', 'I-ExcretorySystem', 'B-TypesOfWaterInBodiesOfWater', 'I-Reptile', 'I-AnimalSystemsProcesses', 'B-Day', 'B-PartsOfTheNervousSystem', 'I-Sky', 'B-Constellation', 'B-MagneticDevice', 'B-TypeOfConsumer', 'I-Minerals', 'I-ProbabilityAndCertainty', 'B-Unit', 'B-Extinction', 'B-Gymnosperm', 'I-ConservationLaws', 'B-AbilityAvailability', 'I-Cycles', 'I-ReleaseEnergy', 'B-BlackHole', 'B-SouthernHemisphereLocations', 'I-DensityUnit', 'I-Soil', 'B-Cause', 'I-CirculatorySystem', 'B-Create', 'B-PrepositionalDirections', 'B-DigestionActions', 'B-PopularMedia', 'I-CelestialEvents', 'B-AnalyzingResearch', 'I-Injuries', 'B-Cycles', 'B-TemperatureMeasuringTools', 'B-DURATION', 'B-AnimalSystemsProcesses', 'B-PhysicalChange', 'I-Angiosperm', 'I-PartsOfTheSkeletalSystem', 'I-PrenatalOrganismStates', 'I-Negations', 'B-ImmuneSystem', 'I-GroupsOfScientists', 'B-TerrestrialLocations', 'B-MeasurementsForHeatChange', 'B-Response', 'I-CookingToolsFood', 'I-Associate', 'I-ScientificTools', 'B-TheoryOfPhysics', 'B-Circuits', 'B-RespiratorySystem', 'I-GeopoliticalLocations', 'I-GaseousMatter', 'B-SolarSystem', 'B-ActionsForAnimals', 'I-Color', 'B-Compound', 'B-TimeMeasuringTools', 'I-Wetness', 'I-BeliefKnowledge', 'I-SolarSystem', 'B-PartsOfTheIntegumentarySystem', 'I-ChemicalProcesses', 'I-InsectAnimalPart', 'I-Use', 'I-Material', 'B-Validity', 'I-MechanicalMovement', 'B-TidesHighTideLowTide', 'B-Bryophyte', 'B-Genetics', 'B-PlantNutrients', 'B-GranularSolids', 'B-PartsOfWaterCycle', 'I-PropertiesOfSickness', 'I-CelestialMeasurements', 'B-Asteroid', 'I-Response', 'B-CellsAndGenetics', 'B-PERCENT', 'I-Permit', 'I-Safety', 'B-Start', 'B-ArcheologicalProcessTechnique', 'B-PartsOfAVirus', 'B-Measurements', 'B-SolidMatter', 'I-Height', 'I-Cities', 'I-BusinessNames', 'I-Aquatic', 'I-SystemParts', 'I-FossilTypesIndexFossil', 'I-GeographicFormationProcess', 'I-PartsOfABusiness', 'I-Extinction', 'I-Adaptation', 'B-EnergyWaves', 'I-Composition', 'B-Source', 'B-Bacteria', 'B-Behaviors', 'I-TypesOfTerrestrialEcosystems', 'B-Reactions', 'B-Indicate', 'B-PropertiesOfSickness', 'I-AtomicProperties', 'I-SeasonsFallAutumnWinterSpringSummer', 'I-PartsOfARepresentation', 'B-TectonicPlates', 'B-Fungi', 'B-Depth', 'B-Meteor', 'I-RelativeDirection', 'I-HeatingAppliance', 'I-Constellation', 'I-RelativeTime', 'B-CookingToolsFood', 'B-WeightMeasuringTool', 'I-GeographicFormations', 'I-Age', 'B-Angiosperm', 'B-CleanUp', 'B-MuscularSystem', 'I-ActionsForNutrition', 'B-ManmadeObjects', 'I-WavePerception', 'I-AquaticAnimalPart', 'B-Aquatic', 'I-AnimalClassificationMethod', 'B-FossilForming', 'B-GeometricMeasurements', 'B-Cost', 'I-DATE', 'B-PoorHealth', 'I-Device', 'B-PhaseChangingActions', 'I-Language', 'I-TypesOfChemicalReactions', 'I-TectonicPlates', 'B-Hardness', 'B-HardnessUnit', 'I-Fungi', 'B-SoundEnergy', 'I-Choose', 'B-Alter', 'B-Frequency', 'I-SoundProducingObject', 'I-Measurements', 'B-PERSON', 'B-PressureMeasuringTool', 'I-PushingForces', 'B-Represent', 'I-PullingForces', 'I-PH', 'B-Rigidity', 'I-ConstructionTools', 'I-NUMBER', 'I-Classify', 'I-TIME', 'I-PartsOfTheRespiratorySystem', 'I-Medicine', 'B-RelativeTime', 'I-Compound', 'B-PartsOfDNA', 'I-ObjectQuantification', 'B-Growth', 'I-SoundEnergy', 'I-ElectricalEnergy', 'I-LiquidMatter', 'B-DATE', 'I-Habitat', 'B-PushingActions', 'B-SpaceMissionsEGApolloGeminiMercury', 'B-Communicate', 'B-Mammal', 'I-Reproduction', 'I-AnimalAdditionalCategories', 'B-MeasuresOfAmountOfLight', 'B-ManmadeLocations', 'B-TypesOfChemicalReactions', 'B-PowerUnit', 'I-Represent', 'B-ChemicalProduct', 'B-Traffic', 'B-Result', 'I-Gravity', 'B-ORGANIZATION', 'I-ParticleMovement', 'I-DistanceMeasuringTools', 'I-PropertiesOfSoil', 'I-MolecularProperties', 'I-RespiratorySystem', 'I-OuterPlanets', 'B-PressureUnit', 'I-LightProducingObject', 'B-ConcludingResearch', 'I-ElectromagneticSpectrum', 'I-GeometricMeasurements', 'I-Separate', 'B-Advertising', 'B-Height', 'B-Collect', 'B-Distance', 'I-TemporalProperty', 'B-TemperatureUnit', 'B-Forests', 'B-PartsOfTheDigestiveSystem', 'B-GenericTerms', 'I-Calculations', 'I-CirculationActions', 'I-TimeUnit', 'I-AtomComponents', 'B-BeliefKnowledge', 'B-BodiesOfWater', 'B-OpportunitiesAndTheirExtent', 'B-TimesOfDayDayNight', 'B-AcidityUnit', 'B-SystemParts', 'B-Meals', 'B-NervousSystem', 'B-Separation', 'I-PartsOfTheImmuneSystem', 'I-Mammal', 'B-Spectra', 'B-Groups', 'I-FormChangingActions', 'I-LocationChangingActions', 'I-MassMeasuringTool', 'B-Plant', 'B-Habitat', 'B-PlanetParts', 'B-FossilRecordTimeline', 'I-PlantPart', 'B-TraceFossil', 'B-AmphibianAnimalPart', 'B-CellProcesses', 'I-CarbonCycle', 'B-VolumeUnit', 'B-SpaceProbes', 'I-Mixtures', 'I-Touch', 'I-OtherOrganismProperties', 'B-TechnologicalInstrument', 'B-Particles', 'B-UnderwaterEcosystem', 'I-Protist', 'B-OtherEnergyResources', 'B-Flammability', 'B-Birth', 'B-CapillaryAction', 'B-Biology', 'B-ExamplesOfHabitats', 'I-PartsOfTheMuscularSystem', 'B-Succeed', 'B-Discovery', 'B-Inheritance', 'B-FiltrationTool', 'B-ElectricalEnergySource', 'B-DigestiveSystem', 'I-Preserve', 'I-Start', 'B-TimeUnit', 'I-EndocrineSystem', 'B-MeteorologicalModels', 'B-FrequencyUnit', 'B-ObservationTechniques', 'B-SeasonsFallAutumnWinterSpringSummer', 'B-EnergyUnit', 'I-Magnetic', 'I-NonlivingPartsOfTheEnvironment', 'B-PropertyOfProduction', 'I-LevelOfInclusion', 'I-Communicate', 'B-ElementalComponents', 'B-WaitStay', 'B-ActUponSomething', 'B-Homeostasis', 'I-PhysicalChange', 'B-SpecificNamedBodiesOfWater', 'B-AbsorbEnergy', 'B-Vacuum', 'I-PartsOfABuilding', 'I-BodiesOfWater', 'I-ChemicalChange', 'I-RespirationActions', 'B-PhaseChanges', 'I-ChemicalProduct', 'B-EcosystemsEnvironment', 'I-PrepositionalDirections', 'B-GeneticProperty', 'I-CelestialObject', 'I-ViewingTools', 'I-TypeOfConsumer', 'I-ApparentCelestialMovement', 'B-Identify', 'B-AtomicProperties', 'I-StopRemove', 'I-OtherGeographicWords', 'I-CoolingAppliance', 'B-ChangeInto', 'B-VariablesControls', 'I-ExamplesOfHabitats', 'I-DistanceComparison', 'B-AreaUnit', 'I-LayersOfTheEarth', 'I-NationalityOrigin', 'I-WaitStay', 'I-PartsOfDNA', 'B-Hypothesizing', 'B-PartsOfObservationInstruments', 'B-PartsOfChemicalReactions', 'B-AquaticAnimalPart', 'B-Senses', 'I-NaturalPhenomena', 'I-ReproductiveSystem', 'B-Sickness', 'B-ViewingTools', 'B-CastFossilMoldFossil', 'B-AtmosphericLayers', 'B-Language', 'B-ExcretoryActions', 'B-Group', 'I-ContainBeComposedOf', 'I-TransferEnergy', 'B-ActionsForNutrition', 'I-Inheritance', 'B-Help', 'I-ArcheologicalProcessTechnique', 'I-AirVehicle', 'B-PhysicalProperty', 'B-Value', 'B-AtomComponents', 'B-PartsOfTheMuscularSystem', 'I-Source', 'I-SystemOfCommunication', 'B-StarLayers', 'B-CombineAdd', 'B-Safety', 'I-PerformingExperimentsWell', 'I-ChangeInComposition', 'B-PropertiesOfSoil', 'B-ExamplesOfSounds', 'B-Patents', 'I-ComputingDevice', 'I-WaterVehicle', 'I-TheUniverseUniverseAndItsParts', 'I-MagneticForce', 'I-Help', 'B-PartsOfTheReproductiveSystem', 'I-OtherHumanProperties', 'I-CapillaryAction', 'B-ContainBeComposedOf', 'I-MeasuringSpeed', 'I-AmountChangingActions', 'I-Human', 'B-WeatherDescriptions', 'I-RelativeLocations', 'B-SyntheticMaterial', 'B-ChemicalProcesses', 'I-Identify', 'I-CleanUp', 'I-NutritiveSubstancesForAnimalsOrPlants', 'B-PushingForces', 'B-ChangeInComposition', 'B-Device', 'I-Discovery', 'B-CoolingAppliance', 'B-Unknown', 'I-Cell', 'I-Rarity', 'I-StateOfBeing', 'B-Use', 'B-InnerPlanets', 'I-Hypothesizing', 'I-EclipseEvents', 'B-SystemOfCommunication', 'I-HardnessUnit', 'I-Eukaryote', 'B-ResistanceStrength', 'B-PullingForces', 'I-LandVehicle', 'I-CardinalNumber', 'I-FiltrationTool', 'B-SkeletalSystem', 'I-AnimalPart', 'B-ArithmeticMeasure', 'B-ScientificTools', 'I-ArithmeticMeasure', 'B-MetalSolids', 'B-SubstancesProducedByPlantProcesses', 'B-Texture', 'B-OtherAnimalProperties', 'B-Transportation', 'I-PartsOfAVirus', 'I-ManmadeLocations', 'B-GaseousMatter', 'I-ProduceEnergy', 'I-MeteorologicalModels', 'B-ReleaseEnergy', 'B-MolecularProperties', 'B-ParticleMovement', 'I-Light', 'B-Preserve', 'I-PhaseChangingActions', 'I-Sickness', 'B-AmountChangingActions', 'B-Adaptation', 'I-TidesHighTideLowTide', 'B-Archaea', 'I-Conductivity', 'I-Rock', 'I-DistanceUnit', 'I-MagneticEnergy', 'I-ElectricalEnergySource', 'B-BehavioralAdaptation', 'I-TemperatureUnit', 'B-Speed', 'B-GeographicFormationProcess', 'B-EndocrineSystem', 'I-LifeCycle', 'I-BacteriaPart', 'I-PlantProcesses', 'B-Death', 'B-PartsOfEarthLayers', 'I-TypesOfIllness', 'I-Locations', 'B-VisualProperty', 'I-Goal', 'B-Cities', 'B-PlantProcesses', 'I-Unknown', 'I-Currents', 'B-PerformingResearch', 'I-SensoryTerms', 'B-MarkersOfTime', 'I-PerformAnActivity', 'I-BusinessIndustry', 'I-ClothesTextiles', 'B-Gene', 'B-LivingDying', 'I-LunarPhases', 'B-EnvironmentalDamageDestruction', 'B-ObjectPart', 'B-Taxonomy', 'B-Divide', 'I-TerrestrialLocations', 'I-Divide', 'I-Groups', 'B-Medicine', 'I-GeneticProcesses', 'B-PartsOfTheImmuneSystem', 'I-Particles', 'B-CardinalNumber', 'B-RelativeLocations', 'B-ElectricityGeneration', 'B-LifeCycle', 'I-Occupation', 'B-ChemicalProperty', 'I-Development', 'I-Result', 'I-FossilForming', 'B-RespirationActions', 'I-Reactions', 'I-Numbers', 'B-QualityComparison', 'B-AvoidReject', 'I-Element', 'B-MechanicalMovement', 'I-Problem', 'B-MammalAnimalPart', 'I-Foods', 'B-NaturalMaterial', 'B-DensityUnit', 'B-WordsForOffspring', 'I-AmountComparison', 'I-MineralProperties', 'B-Mixtures', 'B-PhasesOfWater', 'B-MagneticForce', 'I-GranularSolids', 'I-ObservationInstrumentsTelescopeBinoculars', 'B-PartsOfABusiness', 'B-ChemicalChange', 'B-TIME', 'B-LocationChangingActions', 'B-Believe', 'B-MineralProperties', 'B-OutbreakClassification', 'I-WeightMeasuringTool', 'I-Evolution', 'I-MedicalTerms', 'B-Changes', 'I-CelestialLightOnEarth', 'I-SpacecraftHumanRated', 'B-OtherHumanProperties', 'I-Precipitation', 'B-CirculatorySystem', 'I-CelestialMovement', 'B-GeometricUnit', 'B-LayersOfTheEarth', 'B-PerformingExperimentsWell', 'B-Moon', 'I-StateOfMatter', 'B-TypesOfEvent', 'B-SoundMeasuringTools', 'I-DigestiveSystem', 'B-ExcretorySystem', 'I-ImportanceComparison', 'B-OrganicProcesses', 'B-PropertiesOfWaves', 'B-PropertiesOfFood', 'B-Representation', 'B-SoundProducingObject', 'B-ClothesTextiles', 'I-ExcretoryActions', 'B-Size', 'I-AcademicMedia', 'I-PropertiesOfFood', 'B-PullingActions', 'B-ScientificMethod', 'B-Choose', 'B-RelativeDirection', 'B-SpaceAgencies', 'B-PlantCellPart', 'I-Permeability', 'B-PartsOfBodiesOfWater', 'B-Months', 'B-PartsOfRNA', 'B-Occupation', 'B-Insect', 'I-Mass', 'B-ManMadeGeographicFormations', 'B-Matter', 'O', 'B-MedicalTerms', 'I-FeedbackMechanism', 'B-Buy', 'B-ScientificAssociationsAdministrations', 'B-Directions', 'B-Amphibian', 'I-DigestionActions', 'B-TemporalProperty', 'I-Observe', 'B-NorthernHemisphereLocations', 'B-Nutrition', 'B-Evolution', 'B-PartsOfAChromosome', 'I-Temperature', 'I-PhaseChanges', 'B-Mass', 'I-BlackHole', 'B-BusinessIndustry', 'B-CarbonCycle', 'B-Actions', 'B-Light', 'B-Occur', 'B-CirculationActions', 'B-Precipitation', 'B-Health', 'I-ChangeInto', 'B-Material', 'B-PartsOfTheRespiratorySystem', 'I-HumanPart', 'I-AvoidReject', 'B-MassMeasuringTool', 'B-Soil', 'B-GeologicalEonsErasPeriodsEpochsAges', 'B-MoneyTerms', 'B-Protist', 'I-Rigidity', 'B-InsectAnimalPart', 'I-GeneticRelations', 'I-StarTypes', 'B-Brightness', 'B-VisualComparison', 'I-GeneticProperty', 'B-Consumption', 'B-EndocrineActions', 'B-VerbsForLocate', 'I-Validity', 'B-WordsRelatingToCosmologicalTheoriesExpandContract', 'B-PartsOfTheExcretorySystem', 'B-Planet', 'I-PopularMedia', 'I-SkeletalSystem', 'B-Audiences', 'I-Position', 'B-ObservationPlacesEGObservatory', 'I-StructuralAdaptation', 'B-Permit', 'I-Separation', 'I-Verify', 'B-PercentUnit', 'B-Fossils', 'I-TechnologicalComponent', 'B-LunarPhases', 'I-EnvironmentalPhenomena', 'B-WaterVehiclePart', 'B-ProduceEnergy', 'I-NorthernHemisphereLocations', 'B-CardinalDirectionsNorthEastSouthWest', 'B-Year', 'I-VisualProperty', 'B-Vehicle', 'I-DURATION', 'B-SeparatingMixtures', 'I-SouthernHemisphereLocations', 'B-SystemAndFunctions', 'B-Speciation', 'I-ScientificMethod', 'B-LiquidMatter', 'I-Cost', 'I-PartsOfTheDigestiveSystem', 'B-ConservationLaws', 'B-Require', 'I-ColorChangingActions', 'I-Galaxy', 'I-EndocrineActions', 'B-Property', 'B-Differentiate', 'I-ElectricalProperty', 'B-CelestialMeasurements', 'B-Pressure', 'I-IllnessPreventionCuring', 'B-TechnologicalComponent', 'B-ProbabilityAndCertainty', 'I-VehicularSystemsParts', 'I-Meteorology', 'I-Plant', 'B-Position', 'B-ObservationInstrumentsTelescopeBinoculars', 'I-Nebula', 'B-FoodChain', 'B-PartsOfASolution', 'B-Associate', 'I-Shape', 'I-IncreaseDecrease', 'B-ActionsForAgriculture', 'B-RepresentingElementsAndMolecules', 'I-PartsOfChemicalReactions', 'I-Experimentation', 'I-EnvironmentalDamageDestruction', 'B-TheUniverseUniverseAndItsParts', 'B-LightExaminingTool', 'B-Undiscovered', 'I-EmergencyServices', 'I-WrittenMedia', 'I-Spectra', 'B-AnimalPart', 'B-Wetness', 'B-Eukaryote', 'B-MagneticDirectionMeasuringTool', 'B-Element', 'B-Minerals', 'I-CellsAndGenetics', 'B-ScientificMeetings', 'B-SpacecraftSubsystem', 'I-Geography', 'B-StopRemove', 'I-GalaxyParts', 'B-Verify', 'B-OtherProperties', 'B-PhysicalActivity', 'B-Permeability', 'B-NaturalSelection', 'I-TechnologicalInstrument', 'I-SeparatingMixtures', 'B-Reproduction', 'I-Force', 'B-Sedimentary', 'I-SafetyEquipment', 'B-BusinessNames', 'B-Color', 'B-ElectromagneticSpectrum', 'B-LevelOfInclusion', 'B-LightProducingObject', 'I-WordsForData', 'B-OtherOrganismProperties', 'B-Countries', 'I-NaturalResources', 'B-AnimalAdditionalCategories', 'B-BacteriaPart', 'B-Comparisons', 'I-PartsOfBodiesOfWater', 'B-TypesOfTerrestrialEcosystems', 'I-TheoryOfPhysics', 'I-Biology', 'I-Metabolism', 'B-States', 'B-OtherDescriptionsForPlantsBiennialLeafyEtc', 'I-Bird', 'I-AbilityAvailability', 'I-Energy', 'B-Continents', 'B-Complexity', 'B-Foods', 'B-Harm', 'B-InheritedBehavior', 'I-Fossils', 'B-Stability', 'I-Width', 'I-ThermalEnergy', 'I-Cause', 'I-Require', 'B-LearnedBehavior', 'I-PlanetParts', 'I-MarkersOfTime', 'I-MagneticDevice', 'I-WordsRelatingToCosmologicalTheoriesExpandContract', 'B-ResponseType', 'B-Metamorphic', 'B-Goal', 'I-PullingActions', 'B-ComputingDevice', 'I-MassUnit', 'B-NonlivingPartsOfTheEnvironment', 'B-StateOfMatter', 'I-MetalSolids', 'B-Comet', 'B-SafetyEquipment', 'I-OrganicProcesses', 'I-LiquidHoldingContainersRecepticles', 'B-DwarfPlanets', 'B-Nebula', 'I-Gymnosperm', 'B-CelestialEvents', 'B-SpaceVehicle',
                'B-Classification', "[CLS]", "[SEP]"]
        #return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        # return ["O", 'Buy', 'Mass', 'GeneticProcesses', 'ChangeInLocation', 'AnimalSystemsProcesses', 'QuestionActivityType',
        #         'Evolution', 'PartsOfAGroup', 'EndocrineActions', 'Ability', 'PlantProcesses', 'GalaxyParts', 'InheritedBehavior',
        #         'Reproduction', 'SpeedUnit', 'Forests', 'PERCENT', 'Composition', 'SystemParts', 'HumanPart', 'LiquidMovement',
        #         'TraceFossil', 'LightExaminingTool', 'Growth', 'PartsOfTheMuscularSystem', 'PlantPart', 'OtherProperties',
        #         'AbilityAvailability', 'OrganicCompounds', 'GaseousMatter', 'PartsOfBodiesOfWater', 'DistanceMeasuringTools',
        #         'TrueFormFossil', 'DigestionActions', 'Locations', 'ObjectQuantification', 'CastFossilMoldFossil', 'Negations',
        #         'MammalAnimalPart', 'Associate', 'Material', 'Transportation', 'Satellite', 'Cause', 'GapsAndCracks', 'Pressure',
        #         'Mammal', 'Choose', 'PhaseChanges', 'LiquidMatter', 'Force', 'TimeUnit', 'Surpass', 'Wetness', 'FrequencyUnit',
        #         'PartsOfTheIntegumentarySystem', 'DURATION', 'ExcretoryActions', 'Permeability', 'MarkersOfTime', 'AquaticAnimalPart',
        #         'NUMBER', 'PartsOfTheEye', 'MetalSolids', 'ScientificTools', 'Countries', 'SolarSystem', 'Meteor', 'WeatherPhenomena',
        #         'FeedbackMechanism', 'Continents', 'SystemOfCommunication', 'PartsOfEarthLayers', 'TimeMeasuringTools', 'Archaea',
        #         'PoorHealth', 'ChemicalProduct', 'Constellation', 'Nutrition', 'SpaceVehicle', 'PercentUnit', 'Arachnid', 'TIME', 'Alter',
        #         'OutbreakClassification', 'ThermalEnergy', 'CelestialMovement', 'DATE', 'Comparisons', 'PartsOfTheExcretorySystem',
        #         'NervousSystem', 'OtherAnimalProperties', 'Compete', 'Moon', 'Extinction', 'OtherGeographicWords',
        #         'OtherDescriptionsForPlantsBiennialLeafyEtc', 'Source', 'ChangeInComposition', 'ObservationPlacesEGObservatory',
        #         'ObservationTechniques', 'ElectricalEnergySource', 'Angiosperm', 'StopRemove', 'LOCATION', 'Collect', 'Actions',
        #         'CoolingAppliance', 'WordsForOffspring', 'MassMeasuringTool', 'AbsorbEnergy', 'PartsOfChemicalReactions',
        #         'GeopoliticalLocations', 'AcademicMedia', 'Shape', 'LandVehicle', 'Toxins', 'CardinalDirectionsNorthEastSouthWest',
        #         'PartsOfABusiness', 'PERSON', 'Calculations', 'Event', 'PropertiesOfWaves', 'MeteorologicalModels', 'Gravity',
        #         'UnderwaterEcosystem', 'Gender', 'Spectra', 'TypesOfTerrestrialEcosystems', 'Undiscovered', 'NationalityOrigin',
        #         'Traffic', 'Harm', 'Protist', 'Unknown', 'DigestiveSubstances', 'AmountComparison', 'Patents', 'Difficulty',
        #         'ChangesToResources', 'ScientificAssociationsAdministrations', 'TemperatureUnit', 'PhysicalProperty', 'Speciation',
        #         'SpaceProbes', 'Changes', 'Color', 'PartsOfWaterCycle', 'Viewpoint', 'ElectricityGeneration', 'MineralProperties',
        #         'Group', 'PartsOfAChromosome', 'ManmadeObjects', 'OuterPlanets', 'PropertiesOfSickness', 'AstronomyAeronautics', 'EmergencyServices', 'Consumption', 'VisualComparison', 'ResistanceStrength', 'MagneticForce', 'Products', 'ReleaseEnergy', 'Reptile', 'TheUniverseUniverseAndItsParts', 'Appliance', 'CelestialLightOnEarth', 'VolumeUnit', 'Advertising', 'Metabolism', 'AmountChangingActions', 'PartsOfAVirus', 'PartsOfABuilding', 'Uptake', 'Move', 'Magnetic', 'Circuits', 'ClothesTextiles', 'Rock', 'MechanicalMovement', 'Substances', 'TheoryOfMatter', 'Differentiate', 'ConstructionTools', 'AreaUnit', 'Directions', 'GeologicTheories', 'Amphibian', 'Homeostasis', 'CookingToolsFood', 'Development', 'Property', 'Size', 'PullingActions', 'Foods', 'GeometricSpatialObjects', 'PerformAnActivity', 'SoundProducingObject', 'WeightMeasuringTool', 'CelestialObject', 'RepresentingElementsAndMolecules', 'SpacecraftSubsystem', 'Temperature', 'SensoryTerms', 'SolidMatter', 'PropertiesOfFood', 'Blood', 'SoundMeasuringTools', 'VehicularSystemsParts', 'TechnologicalInstrument', 'Method', 'Help', 'ComputingDevice', 'ManMadeGeographicFormations', 'Representation', 'Release', 'Observe', 'StarTypes', 'SafetyEquipment', 'ScientificMeetings', 'PhasesOfWater', 'PlanetParts', 'TectonicPlates', 'DistanceUnit', 'ContainBeComposedOf', 'BusinessIndustry', 'AirVehicle', 'LifeCycle', 'LevelOfInclusion', 'DistanceComparison', 'PhysicalActivity', 'LivingThing', 'Stability', 'WaitStay', 'SpaceMissionsEGApolloGeminiMercury', 'MuscularSystem', 'Rarity', 'PostnatalOrganismStages', 'EcosystemsEnvironment', 'Months', 'AnimalClassificationMethod', 'ElectricalEnergy', 'RelativeLocations', 'Rigidity', 'TechnologicalComponent', 'PhaseChangingActions', 'Examine', 'Bird', 'Occupation', 'LightMovement', 'ChemicalProperty', 'Meals', 'LocationChangingActions', 'LunarPhases', 'Aquatic', 'TransferEnergy', 'Goal', 'Brightness', 'Indicate', 'PrenatalOrganismStates', 'VisualProperty', 'ORDINAL', 'Death', 'GeometricMeasurements', 'AnimalAdditionalCategories', 'HeatingAppliance', 'PartsOfARepresentation', 'EarthPartsGrossGroundAtmosphere', 'MassUnit', 'Break', 'Value', 'SubstancesProducedByPlantProcesses', 'ObservationInstrumentsTelescopeBinoculars', 'BirdAnimalPart', 'MagneticEnergy', 'OtherOrganismProperties', 'OtherEnergyResources', 'ScientificMethod', 'StarLayers', 'CoolingToolsFood', 'Mutation', 'Validity', 'CellsAndGenetics', 'BodiesOfWater', 'ElementalComponents', 'ActionsForNutrition', 'Height', 'CleanUp', 'Require', 'ExcretorySystem', 'SeedlessVascular', 'Monera', 'Pattern', 'PrepositionalDirections', 'GenericTerms', 'Problem', 'Energy', 'Light', 'Length', 'ScientificTheoryExperimentationAndHistory', 'PushingActions', 'ObjectPart', 'CelestialMeasurements', 'ResponseType', 'PropertyOfMotion', 'MeasuresOfAmountOfLight', 'WrittenMedia', 'Injuries', 'Conductivity', 'ActionsForTides', 'AngleMeasuringTools', 'ElectricAppliance', 'GeneticProperty', 'Employment', 'ConstructiveDestructiveForces', 'Relevant', 'Bacteria', 'Flammability', 'Verify', 'Particles', 'GaseousMovement', 'ActionsForAnimals', 'SkeletalSystem', 'Device', 'Geography', 'Galaxy', 'Discovery', 'Hypothesizing', 'PressureUnit', 'DigestiveSystem', 'Unit', 'Result', 'Permit', 'ConcludingResearch', 'Adaptation', 'TypeOfConsumer', 'FossilFuel', 'IncreaseDecrease', 'Human', 'PartsOfDNA', 'TimesOfDayDayNight', 'NaturalResources', 'Star', 'RelativeDirection', 'MagneticDirectionMeasuringTool', 'Preserve', 'ClassesOfElements', 'ElectricalUnit', 'RespiratorySystem', 'Medicine', 'ChangeInto', 'EclipseEvents', 'Nebula', 'TidesHighTideLowTide', 'Position', 'Identify', 'RelativeNumber', 'MoneyTerms', 'SeasonsFallAutumnWinterSpringSummer', 'GuidelinesAndRules', 'PopularMedia', 'SouthernHemisphereLocations', 'PartsOfRNA', 'Year', 'NaturalMaterial', 'MolecularProperties', 'RespirationActions', 'Mixtures', 'InnerPlanets', 'ExamplesOfSounds', 'Gene', 'SpacecraftHumanRated', 'Represent', 'VerbsForLocate', 'AtomComponents', 'Bryophyte', 'Separation', 'EnergyUnit', 'LearnedBehavior', 'Metamorphic', 'WordsRelatingToCosmologicalTheoriesExpandContract', 'RelativeTime', 'ArithmeticMeasure', 'SpecificNamedBodiesOfWater', 'WordsForData', 'Insect', 'PhysicalChange', 'TemporalProperty', 'Agriculture', 'PlantCellPart', 'NonlivingPartsOfTheEnvironment', 'Divide', 'OrganicProcesses', 'States', 'Meteorology', 'Texture', 'FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'PartsOfTheNervousSystem', 'TerrestrialLocations', 'SimpleMachines', 'ProduceEnergy', 'Igneous', 'AmphibianAnimalPart', 'ElectricalProperty', 'Vacuum', 'Quality', 'Audiences', 'PartsOfTheReproductiveSystem', 'CosmologicalTheoriesBigBangBigCrunch', 'AnimalCellPart', 'Behaviors', 'Language', 'YearNumerals', 'MuscularSystemActions', 'Start', 'Asteroid', 'ImportanceComparison', 'ColorChangingActions', 'BehavioralAdaptation', 'OtherHumanProperties', 'TypesOfWaterInBodiesOfWater', 'LivingDying', 'IllnessPreventionCuring', 'PressureMeasuringTool', 'ImmuneSystem', 'AvoidReject', 'InsectAnimalPart', 'PartsOfObservationInstruments', 'ORGANIZATION', 'Animal', 'Plant', 'SystemAndFunctions', 'PartsOfTheSkeletalSystem', 'Taxonomy', 'ActionsForAgriculture', 'TypesOfEvent', 'GeographicFormationParts', 'ActUponSomething', 'PlantNutrients', 'Currents', 'ReptileAnimalPart', 'Succeed', 'GeographicFormations', 'Sedimentary', 'CardinalNumber', 'FormChangingActions', 'EndocrineSystem', 'Communicate', 'ProbabilityAndCertainty', 'PartsOfTheRespiratorySystem', 'WavePerception', 'Soil', 'Inheritance', 'PropertiesOfSoil', 'BeliefKnowledge', 'WeatherDescriptions', 'AtmosphericLayers', 'TemperatureMeasuringTools', 'ReplicatingResearch', 'PH', 'GeometricUnit', 'SystemProcessStages', 'Touch', 'CelestialEvents', 'Fossils', 'Eukaryote', 'Cities', 'Comet', 'GeologicalEonsErasPeriodsEpochsAges', 'TheoryOfPhysics', 'BlackHole', 'PowerUnit', 'DensityUnit', 'Relations', 'CirculationActions', 'FoodChain', 'PushingForces', 'PartsOfEndocrineSystem', 'GeneticRelations', 'ArcheologicalProcessTechnique', 'ChemicalChange', 'FossilForming', 'Response', 'NutritiveSubstancesForAnimalsOrPlants', 'CapillaryAction', 'ExamplesOfHabitats', 'FossilRecordTimeline', 'ManmadeLocations', 'GroupsOfScientists', 'TypesOfIllness', 'Precipitation', 'Health', 'TypesOfChemicalReactions', 'CombineAdd', 'ChemicalProcesses', 'WaterVehicle', 'LayersOfTheEarth', 'FiltrationTool', 'Use', 'PartsOfTheFoodChain', 'Safety', 'ViewingTools', 'Groups', 'LiquidHoldingContainersRecepticles', 'EnvironmentalPhenomena', 'Day', 'Genetics', 'BusinessNames', 'Classify', 'StateOfMatter', 'ElectricityMeasuringTool', 'CellProcesses', 'Speed', 'Sickness', 'SyntheticMaterial', 'AtomicProperties', 'AcidityUnit', 'Inertia', 'IntegumentarySystem', 'OpportunitiesAndTheirExtent', 'Occur', 'AnalyzingResearch', 'AstronomicalDistanceUnitsLightYearAstronomicalUnitAu', 'Experimentation', 'MeasurementsForHeatChange', 'ApparentCelestialMovement', 'Separate', 'Frequency', 'Matter', 'CarbonCycle', 'MeasuringSpeed', 'Width', 'Birth', 'Fungi', 'QualityComparison', 'SeparatingMixtures', 'Cell', 'VolumeMeasuringTool', 'FossilTypesIndexFossil', 'PartsOfWaves', 'ParticleMovement', 'Vehicle', 'PartsOfTheCirculatorySystem', 'MagneticDevice', 'Numbers', 'MedicalTerms', 'Gymnosperm', 'Depth', 'Believe', 'ElectromagneticSpectrum', 'VariablesControls', 'BacteriaPart', 'Distance', 'Exemplar', 'AnimalPart', 'Habitat', 'PhaseTransitionPoint', 'OrganismRelationships', 'NaturalPhenomena', 'HardnessUnit', 'Complexity', 'Hardness', 'PerformingExperimentsWell', 'ElectricityAndCircuits', 'GranularSolids', 'Compound', 'PullingForces', 'PartsOfTheImmuneSystem', 'StructuralAdaptation', 'PropertyOfProduction', 'SpaceAgencies', 'PartsOfASolution', 'Element', 'LightProducingObject', 'GroupsOfOrganisms', 'Minerals', 'GeographicFormationProcess', 'DwarfPlanets', 'PartsOfTheDigestiveSystem', 'EnvironmentalDamageDestruction', 'Sky', 'WaterVehiclePart', 'Measurements', 'NaturalSelection', 'Cost', 'Reactions', 'Cycles', 'Create', 'EnergyWaves', 'Planet', 'SoundEnergy', 'ConservationLaws', 'Scientists', 'PerformingResearch', 'NorthernHemisphereLocations', 'ForceUnit', 'ResultsOfDecomposition', 'Senses', 'ReproductiveSystem', 'StateOfBeing', 'CirculatorySystem', 'MineralFormations', 'Classification', 'Age',
        #         'Biology', "[CLS]", "[SEP]"]
    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            print('sentence: ',sentence)
            print('label: ',label)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner":NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model,
              from_tf = False,
              config = config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,1)}
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = {i : label for i, label in enumerate(label_list,1)}
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list)+1,"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred,digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)


if __name__ == "__main__":
    main()