from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import pickle

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

from multi_label_ner_performance import classification_report

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

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
        #print('sequence_output: ',sequence_output.shape)
        logits = self.classifier(sequence_output)
        #print('self number labels: ', self.num_labels)
        if labels is not None:
            labels = labels.float()
            loss_fct = nn.BCEWithLogitsLoss()
            #loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                # print('logits.view(-1, self.num_labels): ',logits.view(-1, self.num_labels))
                # print('logits.shape: ', logits.view(-1, self.num_labels).size())
                # print('labels: ',labels.view(-1))
                # print('labels.shape: ', labels.view(-1,self.num_labels).size())
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
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
        label.append([i.strip() for i in splits[4:]])
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
        return ['I-CoolingAppliance', 'B-PropertiesOfFood', 'I-Cost', 'I-AcademicMedia', 'B-Galaxy', 'I-RespirationActions', 'I-Magnetic', 'B-PropertiesOfSoil', 'B-ColorChangingActions', 'I-AnimalPart', 'B-Homeostasis', 'B-Precipitation', 'B-Spectra', 'I-CombineAdd', 'B-OpportunitiesAndTheirExtent', 'B-Size', 'B-Verify', 'B-Matter', 'B-Employment', 'B-NutritiveSubstancesForAnimalsOrPlants', 'I-SafetyEquipment', 'B-ManmadeObjects', 'I-ChemicalChange', 'B-Classification', 'I-NaturalResources', 'I-CarbonCycle', 'B-AbilityAvailability', 'I-MassMeasuringTool', 'I-AreaUnit', 'I-PartsOfTheReproductiveSystem', 'I-AtomComponents', 'B-Actions', 'I-Use', 'B-Response', 'I-Substances', 'I-Viewpoint', 'I-ScientificAssociationsAdministrations', 'B-GeographicFormationProcess', 'B-Extinction', 'B-CellProcesses', 'I-Protist', 'I-CellsAndGenetics', 'B-AvoidReject', 'B-CoolingAppliance', 'I-Fossils', 'I-SystemAndFunctions', 'B-PartsOfEndocrineSystem', 'B-AnimalAdditionalCategories', 'I-PlantNutrients', 'B-Relevant', 'B-RespirationActions', 'I-TheoryOfMatter', 'B-ScientificTools', 'B-SeparatingMixtures', 'B-Property', 'B-FoodChain', 'B-Touch', 'I-Foods', 'I-PropertiesOfFood', 'B-Force', 'I-ArcheologicalProcessTechnique', 'I-IllnessPreventionCuring', 'B-AnimalSystemsProcesses', 'I-Frequency', 'B-Behaviors', 'B-WrittenMedia', 'B-Habitat', 'I-Move', 'I-Uptake', 'I-Start', 'B-Fungi', 'I-GaseousMatter', 'B-Separation', 'B-TheoryOfMatter', 'B-OtherEnergyResources', 'I-AquaticAnimalPart', 'B-Source', 'B-Texture', 'I-Traffic', 'I-Distance', 'B-CirculationActions', 'I-ParticleMovement', 'I-TIME', 'I-PartsOfARepresentation', 'I-Homeostasis', 'I-LandVehicle', 'I-CleanUp', 'I-SpaceAgencies', 'B-DistanceComparison', 'I-ManMadeGeographicFormations', 'B-Reactions', 'B-Adaptation', 'B-RespiratorySystem', 'I-GeneticProcesses', 'B-MolecularProperties', 'I-DensityUnit', 'B-BacteriaPart', 'I-Color', 'B-Locations', 'B-GeometricMeasurements', 'I-Gene', 'I-PartsOfTheDigestiveSystem', 'I-LiquidHoldingContainersRecepticles', 'B-PropertiesOfSickness', 'I-Hypothesizing', 'I-GeneticRelations', 'B-Evolution', 'I-NaturalMaterial', 'I-EmergencyServices', 'B-FossilTypesIndexFossil', 'B-PushingForces', 'B-PartsOfARepresentation', 'B-VisualProperty', 'B-QualityComparison', 'I-Temperature', 'B-Require', 'B-PerformingResearch', 'B-GeometricSpatialObjects', 'B-PartsOfAChromosome', 'I-ElectricalEnergy', 'B-AmountChangingActions', 'B-Sky', 'B-TechnologicalInstrument', 'B-Biology', 'B-PartsOfTheImmuneSystem', 'I-GeneticProperty', 'B-ChemicalChange', 'I-ElectromagneticSpectrum', 'I-TechnologicalComponent', 'I-TimesOfDayDayNight', 'I-PropertiesOfSickness', 'B-TheoryOfPhysics', 'B-ExcretorySystem', 'B-Age', 'B-Substances', 'I-Preserve', 'I-TraceFossil', 'B-ContainBeComposedOf', 'I-StateOfBeing', 'B-PartsOfTheReproductiveSystem', 'B-Observe', 'B-MeasurementsForHeatChange', 'B-EarthPartsGrossGroundAtmosphere', 'I-SolarSystem', 'B-PostnatalOrganismStages', 'B-NUMBER', 'B-ObservationInstrumentsTelescopeBinoculars', 'I-AnimalCellPart', 'B-Traffic', 'B-Harm', 'B-Hypothesizing', 'B-Height', 'B-DURATION', 'B-StateOfBeing', 'B-ReplicatingResearch', 'I-Metabolism', 'I-Identify', 'B-EclipseEvents', 'I-ArithmeticMeasure', 'I-TypesOfTerrestrialEcosystems', 'I-AnimalSystemsProcesses', 'I-RelativeTime', 'I-TypesOfEvent', 'I-TemperatureUnit', 'I-Position', 'I-MagneticForce', 'I-NonlivingPartsOfTheEnvironment', 'B-WeatherPhenomena', 'I-GeographicFormationParts', 'I-SpeedUnit', 'B-TrueFormFossil', 'I-Goal', 'I-Result', 'B-PartsOfEarthLayers', 'B-Audiences', 'I-ExcretoryActions', 'I-RelativeNumber', 'B-Exemplar', 'B-Mammal', 'B-Flammability', 'B-Compound', 'B-Reproduction', 'B-PhysicalChange', 'B-VehicularSystemsParts', 'B-EcosystemsEnvironment', 'I-Reproduction', 'B-BusinessNames', 'I-LivingDying', 'B-BusinessIndustry', 'I-ScientificTools', 'B-LandVehicle', 'B-PullingActions', 'B-PullingForces', 'I-Genetics', 'B-Unknown', 'I-ConcludingResearch', 'B-VolumeMeasuringTool', 'B-AmphibianAnimalPart', 'I-VerbsForLocate', 'B-ComputingDevice', 'I-DistanceMeasuringTools', 'I-ReproductiveSystem', 'I-Constellation', 'I-Precipitation', 'B-ChangeInComposition', 'I-Height', 'B-Result', 'B-MagneticDirectionMeasuringTool', 'B-NaturalMaterial', 'B-Start', 'I-Device', 'I-PhaseChangingActions', 'B-Cause', 'I-Problem', 'B-Divide', 'I-CelestialObject', 'I-Mutation', 'B-CarbonCycle', 'I-Permeability', 'B-States', 'B-LightExaminingTool', 'I-ScientificTheoryExperimentationAndHistory', 'B-ConstructiveDestructiveForces', 'I-Extinction', 'B-Months', 'B-Use', 'I-Collect', 'I-Force', 'B-PartsOfTheEye', 'B-Health', 'I-ChemicalProperty', 'I-ActUponSomething', 'I-MagneticDirectionMeasuringTool', 'B-LightProducingObject', 'B-WeightMeasuringTool', 'I-Plant', 'B-PartsOfABuilding', 'B-CirculatorySystem', 'B-Pressure', 'B-Forests', 'B-MeteorologicalModels', 'I-WaterVehiclePart', 'I-EnvironmentalPhenomena', 'I-TrueFormFossil', 'B-ForceUnit', 'B-SpaceVehicle', 'I-ConstructionTools', 'B-CapillaryAction', 'B-Meals', 'B-Release', 'I-CelestialMovement', 'I-TypesOfIllness', 'B-MagneticForce', 'B-Day', 'B-SpecificNamedBodiesOfWater', 'B-ResultsOfDecomposition', 'B-Differentiate', 'B-Mass', 'I-ObservationInstrumentsTelescopeBinoculars', 'B-InnerPlanets', 'I-ObservationTechniques', 'B-Believe', 'B-ConstructionTools', 'B-ChangesToResources', 'B-Agriculture', 'B-LayersOfTheEarth', 'B-DistanceUnit', 'I-PrenatalOrganismStates', 'I-MolecularProperties', 'B-Color', 'B-Ability', 'B-OuterPlanets', 'I-ElectricityGeneration', 'B-GalaxyParts', 'B-NonlivingPartsOfTheEnvironment', 'B-SpaceProbes', 'B-ORGANIZATION', 'I-Fungi', 'B-NaturalPhenomena', 'B-ArcheologicalProcessTechnique', 'B-Arachnid', 'I-Toxins', 'B-Comet', 'B-Choose', 'I-ProduceEnergy', 'I-MechanicalMovement', 'I-CosmologicalTheoriesBigBangBigCrunch', 'I-ObjectPart', 'I-AnimalAdditionalCategories', 'B-GranularSolids', 'I-ApparentCelestialMovement', 'I-Rarity', 'B-ConcludingResearch', 'I-EcosystemsEnvironment', 'I-SeasonsFallAutumnWinterSpringSummer', 'B-PopularMedia', 'B-Safety', 'I-FossilFuel', 'I-BeliefKnowledge', 'B-CellsAndGenetics', 'B-YearNumerals', 'B-Patents', 'I-Composition', 'I-AbilityAvailability', 'I-Behaviors', 'B-ScientificMeetings', 'I-ContainBeComposedOf', 'I-PartsOfDNA', 'B-PartsOfWaterCycle', 'B-ObjectPart', 'I-Verify', 'I-ChangeInto', 'B-ApparentCelestialMovement', 'B-ProbabilityAndCertainty', 'B-Succeed', 'B-Constellation', 'I-Permit', 'I-VolumeMeasuringTool', 'B-Consumption', 'B-TemperatureMeasuringTools', 'I-ChemicalProcesses', 'B-Star', 'I-HumanPart', 'I-PERCENT', 'I-PartsOfAVirus', 'B-Planet', 'I-Circuits', 'I-Vehicle', 'B-Depth', 'B-Human', 'B-ChemicalProperty', 'I-ElectricalProperty', 'I-Cities', 'I-ExamplesOfHabitats', 'I-OtherGeographicWords', 'B-Inertia', 'B-Stability', 'I-Medicine', 'B-Animal', 'B-MoneyTerms', 'I-AmountComparison', 'B-Metabolism', 'I-GroupsOfScientists', 'B-Nebula', 'I-Eukaryote', 'B-Brightness', 'B-PropertyOfProduction', 'I-SpaceVehicle', 'I-CapillaryAction', 'B-WordsForData', 'I-Negations', 'I-Groups', 'B-TypeOfConsumer', 'I-Source', 'B-Identify', 'B-PhysicalActivity', 'I-TypesOfChemicalReactions', 'I-Response', 'B-Associate', 'B-Complexity', 'B-PlantCellPart', 'B-ObservationTechniques', 'B-OtherProperties', 'B-LevelOfInclusion', 'I-GeologicTheories', 'I-RelativeLocations', 'I-PhysicalActivity', 'I-ChangeInComposition', 'B-Distance', 'I-PerformAnActivity', 'B-PressureUnit', 'B-Alter', 'B-StateOfMatter', 'I-ExcretorySystem', 'B-Light', 'I-Injuries', 'B-ActionsForAnimals', 'B-Soil', 'B-ElectricityAndCircuits', 'B-WaterVehiclePart', 'B-Method', 'B-EndocrineSystem', 'B-PlantPart', 'B-LearnedBehavior', 'I-Bacteria', 'B-GaseousMovement', 'I-Touch', 'I-SpecificNamedBodiesOfWater', 'I-OtherAnimalProperties', 'I-SimpleMachines', 'I-Classify', 'I-Cell', 'I-Discovery', 'I-ChemicalProduct', 'B-ReptileAnimalPart', 'B-CleanUp', 'B-MetalSolids', 'I-Bird', 'B-PartsOfAVirus', 'I-Create', 'I-TidesHighTideLowTide', 'I-Adaptation', 'I-Inheritance', 'B-LivingDying', 'B-AtmosphericLayers', 'B-Mixtures', 'I-StarTypes', 'B-Reptile', 'B-Genetics', 'I-ViewingTools', 'B-ActionsForTides', 'I-PartsOfEndocrineSystem', 'B-LiquidMatter', 'B-Device', 'I-Monera', 'B-Bryophyte', 'B-GroupsOfOrganisms', 'B-Relations', 'I-MassUnit', 'B-Magnetic', 'I-ThermalEnergy', 'I-PrepositionalDirections', 'I-ReleaseEnergy', 'I-ImportanceComparison', 'B-BeliefKnowledge', 'I-NationalityOrigin', 'B-SystemParts', 'I-ResistanceStrength', 'I-Relevant', 'B-CombineAdd', 'B-VariablesControls', 'B-FormChangingActions', 'I-AnalyzingResearch', 'B-TypesOfChemicalReactions', 'I-Quality', 'B-TemporalProperty', 'I-ConstructiveDestructiveForces', 'B-TerrestrialLocations', 'I-InheritedBehavior', 'B-ObjectQuantification', 'B-PartsOfDNA', 'I-LunarPhases', 'B-AtomicProperties', 'B-Minerals', 'B-Communicate', 'B-Frequency', 'B-ManMadeGeographicFormations', 'B-SeedlessVascular', 'B-PartsOfAGroup', 'I-Occur', 'B-Rigidity', 'I-Mammal', 'B-LifeCycle', 'B-Represent', 'I-TimeMeasuringTools', 'B-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'B-SolidMatter', 'B-PartsOfBodiesOfWater', 'I-LightProducingObject', 'B-Angiosperm', 'I-GeologicalEonsErasPeriodsEpochsAges', 'I-AbsorbEnergy', 'I-PH', 'B-ExamplesOfSounds', 'I-SoundEnergy', 'I-MeteorologicalModels', 'I-FormChangingActions', 'B-PerformAnActivity', 'I-Currents', 'B-Permit', 'I-ObjectQuantification', 'B-Currents', 'B-ImmuneSystem', 'B-EnergyUnit', 'B-SubstancesProducedByPlantProcesses', 'B-Satellite', 'B-ResistanceStrength', 'I-PopularMedia', 'I-Consumption', 'I-Relations', 'I-CirculatorySystem', 'B-GuidelinesAndRules', 'B-Element', 'B-ActUponSomething', 'I-Event', 'B-SpacecraftHumanRated', 'B-Gene', 'I-EndocrineSystem', 'I-EnergyUnit', 'I-Compound', 'I-OrganismRelationships', 'I-PropertyOfMotion', 'I-Geography', 'B-AquaticAnimalPart', 'B-ChemicalProcesses', 'I-LearnedBehavior', 'B-OtherAnimalProperties', 'B-IncreaseDecrease', 'B-GeographicFormationParts', 'I-NUMBER', 'B-Gymnosperm', 'B-FeedbackMechanism', 'I-TimeUnit', 'I-Size', 'I-PhaseTransitionPoint', 'I-MedicalTerms', 'I-LiquidMovement', 'B-MuscularSystemActions', 'B-LunarPhases', 'B-Collect', 'B-RelativeLocations', 'B-Meteorology', 'I-TechnologicalInstrument', 'I-CelestialMeasurements', 'B-ParticleMovement', 'I-Angiosperm', 'I-FossilTypesIndexFossil', 'B-NorthernHemisphereLocations', 'I-CardinalNumber', 'B-RelativeDirection', 'I-BacteriaPart', 'I-SouthernHemisphereLocations', 'I-PartsOfTheNervousSystem', 'B-PartsOfRNA', 'I-AnimalClassificationMethod', 'B-PoorHealth', 'I-NorthernHemisphereLocations', 'B-MarkersOfTime', 'B-PartsOfTheSkeletalSystem', 'B-Development', 'B-AirVehicle', 'B-Hardness', 'I-Sickness', 'B-Energy', 'B-Permeability', 'B-CelestialObject', 'I-Numbers', 'I-Nebula', 'B-DwarfPlanets', 'B-Mutation', 'I-MineralFormations', 'B-ResponseType', 'I-Validity', 'B-Vacuum', 'B-MineralFormations', 'I-EnergyWaves', 'I-StateOfMatter', 'B-ClothesTextiles', 'I-Harm', 'I-SoundMeasuringTools', 'B-EmergencyServices', 'B-SystemProcessStages', 'B-StarLayers', 'I-LOCATION', 'B-Gender', 'B-CosmologicalTheoriesBigBangBigCrunch', 'I-Wetness', 'I-Rigidity', 'B-Position', 'B-TransferEnergy', 'B-DigestionActions', 'I-BehavioralAdaptation', 'B-HardnessUnit', 'I-Property', 'B-VisualComparison', 'I-Energy', 'B-EndocrineActions', 'I-GranularSolids', 'I-MarkersOfTime', 'B-Occur', 'I-ColorChangingActions', 'I-PullingActions', 'I-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'B-PhysicalProperty', 'I-WavePerception', 'B-DigestiveSystem', 'B-Unit', 'I-Alter', 'B-Move', 'B-FossilRecordTimeline', 'I-VisualProperty', 'B-MuscularSystem', 'B-DigestiveSubstances', 'B-MassUnit', 'B-Asteroid', 'B-FrequencyUnit', 'B-PlanetParts', 'B-Bird', 'I-PartsOfChemicalReactions', 'I-VolumeUnit', 'B-Fossils', 'I-PartsOfEarthLayers', 'B-TimeMeasuringTools', 'B-Wetness', 'I-SolidMatter', 'B-Rock', 'B-StopRemove', 'B-ElectricalEnergySource', 'I-Believe', 'B-NationalityOrigin', 'B-PartsOfTheFoodChain', 'B-OutbreakClassification', 'I-Unit', 'B-SensoryTerms', 'I-Associate', 'B-Particles', 'I-LivingThing', 'B-SystemAndFunctions', 'B-LivingThing', 'B-SpaceMissionsEGApolloGeminiMercury', 'I-RespiratorySystem', 'B-Taxonomy', 'B-Indicate', 'I-ActionsForTides', 'B-Undiscovered', 'I-Particles', 'I-MagneticDevice', 'B-GeologicTheories', 'B-QuestionActivityType', 'B-CelestialLightOnEarth', 'I-HeatingAppliance', 'I-FossilForming', 'B-VerbsForLocate', 'B-StarTypes', 'B-WordsRelatingToCosmologicalTheoriesExpandContract', 'B-PrepositionalDirections', 'I-MagneticEnergy', 'B-Composition', 'B-SkeletalSystem', 'B-PhasesOfWater', 'B-PressureMeasuringTool', 'I-FoodChain', 'B-PartsOfTheDigestiveSystem', 'I-Continents', 'B-Examine', 'B-LocationChangingActions', 'I-StopRemove', 'B-MagneticDevice', 'I-ElectricAppliance', 'I-PhysicalChange', 'I-SensoryTerms', 'B-Year', 'B-LiquidHoldingContainersRecepticles', 'I-Asteroid', 'I-Speed', 'B-Speed', 'B-MeasuresOfAmountOfLight', 'B-Calculations', 'B-Directions', 'B-Plant', 'I-Reactions', 'B-TheUniverseUniverseAndItsParts', 'B-AcademicMedia', 'B-PhaseTransitionPoint', 'I-GeographicFormationProcess', 'B-OtherGeographicWords', 'B-RelativeNumber', 'B-PercentUnit', 'I-PartsOfTheSkeletalSystem', 'I-Release', 'B-AcidityUnit', 'I-PropertyOfProduction', 'B-Rarity', 'I-OtherHumanProperties', 'I-ORGANIZATION', 'B-ScientificMethod', 'I-Age', 'I-PlantProcesses', 'I-LifeCycle', 'I-PartsOfABusiness', 'I-MineralProperties', 'B-Inheritance', 'B-TectonicPlates', 'I-PullingForces', 'B-AmountComparison', 'I-UnderwaterEcosystem', 'I-DistanceComparison', 'I-Stability', 'B-Toxins', 'B-BodiesOfWater', 'I-TypeOfConsumer', 'B-MineralProperties', 'I-WeatherPhenomena', 'B-WaterVehicle', 'I-Sky', 'I-GeographicFormations', 'B-AstronomicalDistanceUnitsLightYearAstronomicalUnitAu', 'B-SafetyEquipment', 'I-Occupation', 'I-ScientificMethod', 'B-InsectAnimalPart', 'B-OrganicCompounds', 'B-PartsOfABusiness', 'B-PartsOfTheCirculatorySystem', 'B-Cycles', 'B-Compete', 'I-Separation', 'B-Value', 'B-Surpass', 'I-GroupsOfOrganisms', 'I-TheoryOfPhysics', 'B-Buy', 'B-ReleaseEnergy', 'I-Soil', 'B-WordsForOffspring', 'B-EnvironmentalPhenomena', 'B-SyntheticMaterial', 'I-ManmadeLocations', 'B-TypesOfTerrestrialEcosystems', 'I-Spectra', 'I-AvoidReject', 'B-CelestialMeasurements', 'B-ElectricityGeneration', 'B-ManmadeLocations', 'B-ORDINAL', 'B-OtherHumanProperties', 'B-ChangeInto', 'I-Examine', 'B-PartsOfTheIntegumentarySystem', 'B-PartsOfWaves', 'I-OtherEnergyResources', 'B-ActionsForNutrition', 'B-Language', 'B-SpaceAgencies', 'I-Material', 'I-SystemParts', 'I-EarthPartsGrossGroundAtmosphere', 'B-Changes', 'I-Help', 'B-Separate', 'I-Blood', 'B-SoundMeasuringTools', 'I-ImmuneSystem', 'I-PartsOfTheMuscularSystem', 'I-Igneous', 'B-UnderwaterEcosystem', 'B-SeasonsFallAutumnWinterSpringSummer', 'I-TheUniverseUniverseAndItsParts', 'B-Classify', 'B-GeneticProcesses', 'B-CelestialEvents', 'B-SouthernHemisphereLocations', 'I-CirculationActions', 'B-CardinalDirectionsNorthEastSouthWest', 'I-GeopoliticalLocations', 'B-Experimentation', 'I-PowerUnit', 'B-AngleMeasuringTools', 'B-SolarSystem', 'I-TransferEnergy', 'B-RepresentingElementsAndMolecules', 'I-MuscularSystem', 'B-ElementalComponents', 'B-ElectricalEnergy', 'B-Vehicle', 'O', 'B-PartsOfTheNervousSystem', 'B-TypesOfWaterInBodiesOfWater', 'B-WeatherDescriptions', 'I-TectonicPlates', 'B-NaturalSelection', 'I-Communicate', 'B-Cost', 'I-CastFossilMoldFossil', 'I-Nutrition', 'B-ExcretoryActions', 'I-SpaceMissionsEGApolloGeminiMercury', 'I-Forests', 'I-PartsOfTheCirculatorySystem', 'I-Gymnosperm', 'I-LevelOfInclusion', 'B-NervousSystem', 'B-SimpleMachines', 'B-SoundEnergy', 'B-PH', 'B-Material', 'B-HeatingAppliance', 'I-VisualComparison', 'I-TypesOfWaterInBodiesOfWater', 'I-Divide', 'B-PhaseChanges', 'B-TimeUnit', 'I-MoneyTerms', 'B-BirdAnimalPart', 'I-FossilRecordTimeline', 'I-ElectricityAndCircuits', 'B-Numbers', 'B-ScientificAssociationsAdministrations', 'I-TemporalProperty', 'B-ScientificTheoryExperimentationAndHistory', 'B-Birth', 'I-Aquatic', 'B-GeopoliticalLocations', 'I-ClassesOfElements', 'B-PerformingExperimentsWell', 'I-ComputingDevice', 'I-IncreaseDecrease', 'I-OrganicCompounds', 'B-Injuries', 'I-OtherOrganismProperties', 'I-EclipseEvents', 'I-Sedimentary', 'B-Pattern', 'I-PlanetParts', 'B-PushingActions', 'B-CelestialMovement', 'I-WordsForData', 'I-Locations', 'I-WrittenMedia', 'I-ElementalComponents', 'I-LiquidMatter', 'B-GeometricUnit', 'B-ImportanceComparison', 'B-GaseousMatter', 'B-Problem', 'B-ChemicalProduct', 'B-GroupsOfScientists', 'B-NaturalResources', 'I-Human', 'B-Gravity', 'B-AnimalPart', 'B-PartsOfTheRespiratorySystem', 'B-ArithmeticMeasure', 'I-HardnessUnit', 'I-CellProcesses', 'B-PropertyOfMotion', 'B-OtherDescriptionsForPlantsBiennialLeafyEtc', 'I-Minerals', 'I-PartsOfTheExcretorySystem', 'B-ThermalEnergy', 'B-DensityUnit', 'B-ElectricAppliance', 'B-FossilFuel', 'B-ProduceEnergy', 'B-RelativeTime', 'I-Cause', 'I-StructuralAdaptation', 'B-DATE', 'B-Break', 'I-NaturalSelection', 'B-Event', 'I-Exemplar', 'B-Growth', 'I-Calculations', 'I-IntegumentarySystem', 'B-LOCATION', 'B-TypesOfIllness', 'B-MedicalTerms', 'I-ElectricalUnit', 'I-SpacecraftHumanRated', 'I-EndocrineActions', 'B-MeasuringSpeed', 'B-PartsOfTheMuscularSystem', 'I-StarLayers', 'I-ClothesTextiles', 'I-SystemOfCommunication', 'I-Development', 'B-PartsOfChemicalReactions', 'B-PowerUnit', 'B-ElectricalUnit', 'B-IllnessPreventionCuring', 'B-Continents', 'B-LightMovement', 'I-Observe', 'B-Foods', 'B-Medicine', 'B-Meteor', 'B-CoolingToolsFood', 'I-PhysicalProperty', 'I-Habitat', 'B-AnimalCellPart', 'B-PrenatalOrganismStates', 'B-Circuits', 'I-Star', 'B-PERCENT', 'I-Meteorology', 'B-GeneticRelations', 'I-Insect', 'I-PartsOfBodiesOfWater', 'B-FiltrationTool', 'B-Difficulty', 'B-Insect', 'I-BlackHole', 'I-ManmadeObjects', 'I-BodiesOfWater', 'B-Groups', 'I-MetalSolids', 'I-States', 'B-AreaUnit', 'I-GeometricMeasurements', 'B-Sedimentary', 'B-GenericTerms', 'B-Discovery', 'I-DistanceUnit', 'B-Nutrition', 'B-MassMeasuringTool', 'B-MagneticEnergy', 'I-PerformingExperimentsWell', 'I-Reptile', 'I-RelativeDirection', 'B-MammalAnimalPart', 'B-ViewingTools', 'B-ExamplesOfHabitats', 'B-Transportation', 'B-IntegumentarySystem', 'B-CookingToolsFood', 'I-VehicularSystemsParts', 'B-TimesOfDayDayNight', 'B-AbsorbEnergy', 'B-Help', 'B-Validity', 'B-PartsOfTheExcretorySystem', 'B-CastFossilMoldFossil', 'I-Mixtures', 'I-NervousSystem', 'B-WavePerception', 'B-Bacteria', 'B-ActionsForAgriculture', 'I-PERSON', 'I-Experimentation', 'B-PERSON', 'B-Eukaryote', 'I-Element', 'B-Conductivity', 'B-Group', 'B-Archaea', 'I-Senses', 'B-GeneticProperty', 'B-SoundProducingObject', 'B-PlantProcesses', 'I-PartsOfTheRespiratorySystem', 'B-InheritedBehavior', 'B-CardinalNumber', 'B-PlantNutrients', 'I-MeasurementsForHeatChange', 'B-HumanPart', 'B-Representation', 'I-MuscularSystemActions', 'I-PhasesOfWater', 'I-RepresentingElementsAndMolecules', 'I-TerrestrialLocations', 'B-SpeedUnit', 'I-SpaceProbes', 'I-AstronomicalDistanceUnitsLightYearAstronomicalUnitAu', 'I-DigestiveSubstances', 'I-FiltrationTool', 'I-Countries', 'B-GapsAndCracks', 'I-PhaseChanges', 'I-LightMovement', 'B-OrganismRelationships', 'B-Scientists', 'I-PlantCellPart', 'B-ElectricityMeasuringTool', 'B-TechnologicalComponent', 'B-PropertiesOfWaves', 'I-PartsOfABuilding', 'B-Measurements', 'I-Amphibian', 'B-Length', 'B-TidesHighTideLowTide', 'I-BusinessNames', 'I-Scientists', 'B-EnvironmentalDamageDestruction', 'I-DigestiveSystem', 'I-OrganicProcesses', 'B-LiquidMovement', 'I-ResultsOfDecomposition', 'B-MechanicalMovement', 'I-ExamplesOfSounds', 'I-DATE', 'I-Comet', 'B-Goal', 'B-VolumeUnit', 'B-PartsOfASolution', 'B-Temperature', 'B-Moon', 'B-StructuralAdaptation', 'B-AnimalClassificationMethod', 'I-PartsOfTheImmuneSystem', 'I-Galaxy', 'I-ChangeInLocation', 'I-WaterVehicle', 'B-Occupation', 'B-Sickness', 'I-AtomicProperties', 'B-Aquatic', 'B-WaitStay', 'B-Advertising', 'B-FossilForming', 'I-OuterPlanets', 'B-PartsOfObservationInstruments', 'I-NaturalPhenomena', 'I-EnvironmentalDamageDestruction', 'I-WeightMeasuringTool', 'B-PhaseChangingActions', 'I-NutritiveSubstancesForAnimalsOrPlants', 'B-SpacecraftSubsystem', 'I-CookingToolsFood', 'I-VariablesControls', 'B-SystemOfCommunication', 'B-TIME', 'B-TemperatureUnit', 'B-BehavioralAdaptation', 'I-Shape', 'B-OtherOrganismProperties', 'I-Require', 'I-ActionsForAnimals', 'B-Monera', 'B-Negations', 'I-PlantPart', 'I-LayersOfTheEarth', 'B-ElectricalProperty', 'I-Choose', 'I-GalaxyParts', 'I-PartsOfObservationInstruments', 'B-Uptake', 'B-Preserve', 'B-Cell', 'I-SkeletalSystem', 'I-Gravity', 'B-Viewpoint', 'I-Light', 'B-AtomComponents', 'B-AnalyzingResearch', 'B-Comparisons', 'I-FeedbackMechanism', 'B-Quality', 'B-GeologicalEonsErasPeriodsEpochsAges', 'B-GeographicFormations', 'B-Death', 'B-ChangeInLocation', 'I-PerformingResearch', 'I-Evolution', 'I-Appliance', 'B-Create', 'I-PushingForces', 'I-AtmosphericLayers', 'B-Geography', 'I-DURATION', 'I-Pressure', 'B-Width', 'B-BlackHole', 'B-TypesOfEvent', 'B-Blood', 'I-DigestionActions', 'B-Appliance', 'B-Cities', 'I-Separate', 'I-WaitStay', 'I-Comparisons', 'I-ScientificMeetings', 'I-CelestialLightOnEarth', 'I-Cycles', 'I-BirdAnimalPart', 'B-Speciation', 'I-LocationChangingActions', 'I-Conductivity', 'B-OrganicProcesses', 'B-Amphibian', 'B-Protist', 'I-Language', 'B-Products', 'B-ElectromagneticSpectrum', 'I-Biology', 'I-Taxonomy', 'B-Igneous', 'I-Rock', 'I-Representation', 'B-EnergyWaves', 'B-Senses', 'B-Shape', 'B-Metamorphic', 'B-DistanceMeasuringTools', 'B-ReproductiveSystem', 'I-Agriculture', 'B-ClassesOfElements', 'B-Countries', 'I-ElectricalEnergySource', 'B-TraceFossil',
                'I-Mass', "[CLS]", "[SEP]"]
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
            # print('sentence: ',sentence)
            # print('label: ',label)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,0)}
    #print(label_map['I-Wetness'])
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
        #label_ids.append(label_map["[CLS]"])
        label_ids.append(np.array([int(i==label_map["[CLS]"]) for i in range(len(label_list))]).tolist())
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                #label_ids.append(label_map[labels[i]])
                temp_one_hop = np.zeros(len(label_list))
                for item in labels[i]:
                    temp_one_hop[label_map[item]]=1
                label_ids.append(temp_one_hop.tolist())
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        #label_ids.append(label_map["[SEP]"])
        label_ids.append(np.array([int(i==label_map["[SEP]"]) for i in range(len(label_list))]).tolist())
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            #label_ids.append(0)
            label_ids.append(np.zeros(len(label_list)).tolist())
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            #label_ids.append(0)
            label_ids.append(np.zeros(len(label_list)).tolist())
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        #print('label_ids: ', np.array(label_ids).shape)
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
        #print('labels id shape: ', np.array(label_ids).shape)
        #print('labels id: ',label_ids)
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
                        default=64,
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
    num_labels = len(label_list)

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
        #print('all_label_ids shape: ', all_label_ids.shape)
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
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list),"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        #print('all_label_ids shape: ', all_label_ids.shape)
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
        label_map = {i : label for i, label in enumerate(label_list,0)}
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            #print('label_ids', label_ids)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

            #logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            #print('logits: ', logits.shape)
            logits = F.sigmoid(logits)
            #print('logits: ',logits.shape)
            logits = logits.detach().cpu().numpy()
            #print('max logits: ',np.max(logits,axis=2))
            final_logits = (logits>=0.2).astype(int)
            #print('final logits: ',final_logits)
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                flag = 0
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif flag == len(label_map)-1:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_label_ids_list = []
                        temp_logits_list = []
                        for label_temp_idx, label_temp in enumerate(label_ids[i][j]):
                            if int(label_temp) == 1:
                                flag = label_temp_idx
                                if flag == len(label_map)-1:
                                    break
                                temp_label_ids_list.append(label_map[label_temp_idx])
                        for logit_temp_idx, logits_temp in enumerate(final_logits[i][j]):
                            if flag == len(label_map)-1:
                                    break
                            #print('*************************************************')
                            if int(logits_temp)==1:
                                #print(label_map[logit_temp_idx])
                                temp_logits_list.append(label_map[logit_temp_idx])
                        if flag != len(label_map)-1:
                            temp_1.append(temp_label_ids_list)
                            temp_2.append(temp_logits_list)
        #print('y_pred: ', y_pred)
        #print('y_true: ', y_true)
        pickle_dump_large_file(y_true, 'y_true_dev.pkl')
        pickle_dump_large_file(y_pred, 'y_pred_dev.pkl')
        report = classification_report(y_true, y_pred,digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)


if __name__ == "__main__":
    main()