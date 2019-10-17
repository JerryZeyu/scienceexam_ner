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
        return ['B-ReptileAnimalPart', 'B-TypesOfEvent', 'B-Geography', 'B-Genetics', 'B-PartsOfASolution', 'B-MeasuringSpeed', 'B-Use', 'B-AnimalPart', 'I-CelestialLightOnEarth', 'I-RepresentingElementsAndMolecules', 'I-Alter', 'B-Planet', 'B-PrenatalOrganismStates', 'I-Circuits', 'I-Separation', 'I-StructuralAdaptation', 'B-ScientificTools', 'B-Require', 'B-OpportunitiesAndTheirExtent', 'I-ExcretorySystem', 'I-VolumeMeasuringTool', 'B-Meals', 'I-FormChangingActions', 'I-Distance', 'B-AcademicMedia', 'I-RespirationActions', 'I-Comparisons', 'B-Blood', 'B-Consumption', 'B-Development', 'B-Metabolism', 'I-TraceFossil', 'I-ChangeInto', 'I-Event', 'I-VisualProperty', 'B-Method', 'B-GroupsOfScientists', 'B-ResistanceStrength', 'I-ContainBeComposedOf', 'I-Soil', 'I-ProduceEnergy', 'I-EndocrineSystem', 'B-PartsOfTheRespiratorySystem', 'I-PrepositionalDirections', 'I-StarTypes', 'I-Frequency', 'I-ColorChangingActions', 'B-Height', 'I-GeneticProcesses', 'B-Insect', 'B-ReplicatingResearch', 'B-PowerUnit', 'B-Archaea', 'B-CardinalNumber', 'B-Temperature', 'I-Start', 'I-OtherGeographicWords', 'B-ForceUnit', 'B-Observe', 'I-Galaxy', 'B-PartsOfDNA', 'B-FossilFuel', 'B-OtherGeographicWords', 'B-PhasesOfWater', 'I-Occupation', 'I-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'I-PartsOfTheCirculatorySystem', 'B-IntegumentarySystem', 'B-FoodChain', 'I-Substances', 'I-Biology', 'B-OrganismRelationships', 'B-LifeCycle', 'I-Use', 'I-HardnessUnit', 'I-PartsOfTheExcretorySystem', 'B-CellProcesses', 'I-ORGANIZATION', 'I-ReproductiveSystem', 'B-Represent', 'B-PropertiesOfSickness', 'B-NaturalPhenomena', 'I-IntegumentarySystem', 'I-WaterVehiclePart', 'I-Blood', 'I-TheUniverseUniverseAndItsParts', 'B-Reptile', 'B-Employment', 'I-RelativeNumber', 'B-RelativeNumber', 'I-MeasurementsForHeatChange', 'B-CelestialEvents', 'B-Relevant', 'B-PlanetParts', 'B-SoundMeasuringTools', 'I-WaitStay', 'B-Protist', 'I-LightMovement', 'B-WeightMeasuringTool', 'B-OuterPlanets', 'B-Problem', 'B-ElectricityGeneration', 'I-TimeUnit', 'B-ManmadeObjects', 'I-DigestiveSystem', 'B-ActionsForTides', 'B-Occupation', 'B-ArcheologicalProcessTechnique', 'B-Injuries', 'B-Separate', 'I-Inheritance', 'I-Bird', 'B-Frequency', 'B-Stability', 'I-Injuries', 'B-CombineAdd', 'I-Require', 'I-VisualComparison', 'B-PERSON', 'B-SimpleMachines', 'B-Measurements', 'B-TheUniverseUniverseAndItsParts', 'I-OrganicProcesses', 'B-Depth', 'B-SystemParts', 'I-Height', 'B-Calculations', 'B-SpaceVehicle', 'I-AmountChangingActions', 'B-AnimalCellPart', 'I-AcademicMedia', 'I-ManmadeLocations', 'B-PartsOfARepresentation', 'B-Traffic', 'B-GeneticProperty', 'B-PartsOfBodiesOfWater', 'I-Create', 'I-PhysicalChange', 'B-Moon', 'B-EndocrineSystem', 'I-ElectricalEnergySource', 'B-PartsOfTheDigestiveSystem', 'B-Year', 'B-Indicate', 'B-MolecularProperties', 'I-Exemplar', 'B-ElectricityAndCircuits', 'B-OrganicProcesses', 'I-MassUnit', 'B-PERCENT', 'B-CapillaryAction', 'B-VolumeMeasuringTool', 'B-Directions', 'B-Eukaryote', 'B-PartsOfTheExcretorySystem', 'B-Communicate', 'B-SubstancesProducedByPlantProcesses', 'I-WaterVehicle', 'I-AtomComponents', 'I-PartsOfTheSkeletalSystem', 'B-PlantCellPart', 'B-ConstructiveDestructiveForces', 'B-DistanceComparison', 'B-Material', 'B-SolidMatter', 'I-FeedbackMechanism', 'I-Calculations', 'I-SensoryTerms', 'I-DensityUnit', 'I-EclipseEvents', 'I-Meteorology', 'I-Sickness', 'B-TimeMeasuringTools', 'I-Conductivity', 'B-Minerals', 'B-SouthernHemisphereLocations', 'B-TemperatureMeasuringTools', 'B-RelativeLocations', 'I-Harm', 'B-InnerPlanets', 'B-Conductivity', 'B-ActUponSomething', 'I-Minerals', 'B-PartsOfWaves', 'I-Examine', 'I-Vehicle', 'B-SeasonsFallAutumnWinterSpringSummer', 'I-Verify', 'B-AmountComparison', 'B-ScientificTheoryExperimentationAndHistory', 'B-Actions', 'I-PerformingResearch', 'B-CelestialMeasurements', 'B-ChemicalProduct', 'B-TypeOfConsumer', 'I-PlantProcesses', 'I-Agriculture', 'I-CarbonCycle', 'B-Goal', 'B-ActionsForAgriculture', 'I-SystemAndFunctions', 'I-Occur', 'B-Energy', 'I-StateOfMatter', 'I-Property', 'I-ConcludingResearch', 'I-Validity', 'B-Continents', 'B-Divide', 'I-EnergyUnit', 'B-MagneticDirectionMeasuringTool', 'B-ParticleMovement', 'I-Nutrition', 'I-ViewingTools', 'I-OuterPlanets', 'I-EndocrineActions', 'I-ScientificTools', 'B-TechnologicalInstrument', 'I-PropertyOfProduction', 'B-Uptake', 'B-FossilRecordTimeline', 'I-TectonicPlates', 'B-ResponseType', 'B-Mutation', 'I-OrganicCompounds', 'I-Currents', 'I-PH', 'B-MagneticDevice', 'I-BodiesOfWater', 'I-LearnedBehavior', 'B-Fossils', 'I-NUMBER', 'B-TheoryOfMatter', 'B-BacteriaPart', 'I-MagneticEnergy', 'B-Size', 'B-Color', 'B-LunarPhases', 'B-AngleMeasuringTools', 'I-ObservationTechniques', 'I-CirculatorySystem', 'B-ActionsForAnimals', 'B-TransferEnergy', 'B-Products', 'B-Plant', 'B-Appliance', 'I-PartsOfARepresentation', 'B-ElectricalEnergySource', 'B-VisualComparison', 'B-Countries', 'I-VehicularSystemsParts', 'B-WaterVehiclePart', 'I-Element', 'I-Continents', 'I-Pressure', 'B-Separation', 'B-GranularSolids', 'B-MassMeasuringTool', 'I-MeasuringSpeed', 'I-Device', 'B-LevelOfInclusion', 'I-Position', 'B-AirVehicle', 'B-Quality', 'I-TerrestrialLocations', 'I-ManmadeObjects', 'I-Constellation', 'B-SkeletalSystem', 'B-ThermalEnergy', 'I-ClothesTextiles', 'I-Measurements', 'B-IllnessPreventionCuring', 'B-Rock', 'I-DURATION', 'B-GeologicalEonsErasPeriodsEpochsAges', 'I-TemperatureMeasuringTools', 'B-Inheritance', 'I-Metabolism', 'I-AtomicProperties', 'B-PerformingExperimentsWell', 'B-SpaceAgencies', 'I-Consumption', 'B-TidesHighTideLowTide', 'I-SpecificNamedBodiesOfWater', 'B-WeatherPhenomena', 'I-Comet', 'I-LiquidMatter', 'I-Insect', 'B-OtherEnergyResources', 'B-Spectra', 'B-Extinction', 'I-FossilTypesIndexFossil', 'B-PoorHealth', 'I-GeographicFormationParts', 'B-Groups', 'I-MagneticForce', 'I-PartsOfChemicalReactions', 'I-TheoryOfPhysics', 'I-LevelOfInclusion', 'I-CosmologicalTheoriesBigBangBigCrunch', 'I-ObservationInstrumentsTelescopeBinoculars', 'B-SeedlessVascular', 'B-VariablesControls', 'I-Representation', 'B-Ability', 'I-Rarity', 'B-PartsOfTheReproductiveSystem', 'I-Foods', 'B-Device', 'B-StructuralAdaptation', 'I-Represent', 'I-SpacecraftHumanRated', 'B-Mammal', 'B-PartsOfTheEye', 'B-Safety', 'B-Aquatic', 'B-PartsOfRNA', 'I-AvoidReject', 'I-RelativeLocations', 'B-Mass', 'I-ChangeInComposition', 'B-Mixtures', 'I-AreaUnit', 'I-TemporalProperty', 'I-OtherEnergyResources', 'B-NaturalResources', 'B-Hardness', 'B-SpacecraftHumanRated', 'B-ManMadeGeographicFormations', 'B-CarbonCycle', 'B-Differentiate', 'B-EcosystemsEnvironment', 'B-Magnetic', 'I-GeneticProperty', 'B-ObjectQuantification', 'I-ActUponSomething', 'I-MechanicalMovement', 'B-FeedbackMechanism', 'I-Cities', 'B-FiltrationTool', 'I-AnimalSystemsProcesses', 'B-StarTypes', 'B-Occur', 'B-Gymnosperm', 'I-GeographicFormations', 'B-Difficulty', 'I-HeatingAppliance', 'B-ElementalComponents', 'B-Identify', 'I-OtherAnimalProperties', 'I-PartsOfEarthLayers', 'I-LiquidHoldingContainersRecepticles', 'B-LiquidMovement', 'I-Adaptation', 'I-ElectricAppliance', 'B-WeatherDescriptions', 'I-ChemicalProperty', 'I-VolumeUnit', 'I-Scientists', 'I-GeologicalEonsErasPeriodsEpochsAges', 'B-Numbers', 'B-PartsOfAChromosome', 'B-BusinessNames', 'B-ArithmeticMeasure', 'I-OrganismRelationships', 'I-MeteorologicalModels', 'B-Advertising', 'I-CookingToolsFood', 'B-GaseousMovement', 'B-Medicine', 'B-Substances', 'B-GeopoliticalLocations', 'B-ProduceEnergy', 'B-Property', 'B-SoundEnergy', 'I-Mammal', 'I-ObservationPlacesEGObservatory', 'I-DistanceComparison', 'B-PressureMeasuringTool', 'I-WrittenMedia', 'B-QualityComparison', 'B-AtomComponents', 'B-Help', 'B-BodiesOfWater', 'I-GaseousMatter', 'I-Help', 'B-ReproductiveSystem', 'B-GroupsOfOrganisms', 'I-TechnologicalComponent', 'I-Stability', 'B-Element', 'B-PartsOfAGroup', 'B-ObservationInstrumentsTelescopeBinoculars', 'I-MuscularSystem', 'B-CelestialLightOnEarth', 'I-PlantPart', 'I-Age', 'I-BirdAnimalPart', 'B-EclipseEvents', 'B-Cause', 'B-MeasurementsForHeatChange', 'B-MineralProperties', 'B-FossilTypesIndexFossil', 'B-PhaseChangingActions', 'I-Language', 'B-TypesOfTerrestrialEcosystems', 'I-Spectra', 'B-Igneous', 'I-TrueFormFossil', 'B-RelativeDirection', 'B-AvoidReject', 'I-ChangeInLocation', 'B-WaitStay', 'I-FoodChain', 'I-PartsOfDNA', 'I-MetalSolids', 'B-VerbsForLocate', 'I-Unit', 'B-Response', 'B-GeometricSpatialObjects', 'I-TypesOfWaterInBodiesOfWater', 'I-PrenatalOrganismStates', 'B-CosmologicalTheoriesBigBangBigCrunch', 'I-AmountComparison', 'I-SkeletalSystem', 'B-PartsOfTheSkeletalSystem', 'B-Scientists', 'B-PartsOfEarthLayers', 'B-GapsAndCracks', 'B-ManmadeLocations', 'I-ApparentCelestialMovement', 'I-Locations', 'B-OtherHumanProperties', 'B-ProbabilityAndCertainty', 'I-Experimentation', 'I-FossilFuel', 'B-BehavioralAdaptation', 'I-Star', 'B-StopRemove', 'I-Move', 'I-FossilRecordTimeline', 'I-BehavioralAdaptation', 'I-LOCATION', 'B-Choose', 'I-TypesOfEvent', 'I-Permeability', 'I-GeopoliticalLocations', 'B-ConcludingResearch', 'I-Plant', 'I-CombineAdd', 'I-Separate', 'I-ExamplesOfHabitats', 'B-TheoryOfPhysics', 'I-TidesHighTideLowTide', 'I-PopularMedia', 'B-SpecificNamedBodiesOfWater', 'I-ConservationLaws', 'I-Bacteria', 'I-TheoryOfMatter', 'I-PropertiesOfFood', 'I-Classify', 'I-Communicate', 'I-Relevant', 'B-PrepositionalDirections', 'B-Months', 'B-Arachnid', 'I-Composition', 'I-NaturalMaterial', 'B-MeteorologicalModels', 'I-WordsRelatingToCosmologicalTheoriesExpandContract', 'I-ScientificMethod', 'B-States', 'I-Touch', 'I-GeometricMeasurements', 'B-LiquidMatter', 'B-Taxonomy', 'B-WrittenMedia', 'I-BusinessIndustry', 'B-WavePerception', 'I-ObjectQuantification', 'I-Size', 'B-TectonicPlates', 'B-Classification', 'I-RelativeTime', 'B-AnimalAdditionalCategories', 'B-Cost', 'I-SpaceMissionsEGApolloGeminiMercury', 'I-PullingForces', 'I-MolecularProperties', 'I-Force', 'I-Source', 'I-DigestionActions', 'I-CelestialMovement', 'B-GalaxyParts', 'I-Precipitation', 'I-Quality', 'B-Habitat', 'B-TechnologicalComponent', 'B-PropertyOfMotion', 'B-SpaceProbes', 'B-PropertiesOfSoil', 'B-CastFossilMoldFossil', 'I-WeatherPhenomena', 'B-MechanicalMovement', 'B-FormChangingActions', 'B-WaterVehicle', 'I-CardinalNumber', 'B-Gender', 'I-Mixtures', 'I-ThermalEnergy', 'B-Permeability', 'B-PlantPart', 'B-PhysicalChange', 'I-Animal', 'I-Gravity', 'I-PropertiesOfSoil', 'B-HardnessUnit', 'B-AnalyzingResearch', 'I-Response', 'B-MuscularSystemActions', 'B-ORDINAL', 'I-PartsOfABusiness', 'B-TrueFormFossil', 'I-Safety', 'B-PartsOfAVirus', 'B-BusinessIndustry', 'I-Length', 'I-MagneticDirectionMeasuringTool', 'I-PlantCellPart', 'B-NaturalSelection', 'B-CelestialMovement', 'I-Planet', 'I-Choose', 'B-Composition', 'I-BeliefKnowledge', 'B-ChangeInto', 'B-Cities', 'B-TypesOfChemicalReactions', 'I-AnimalCellPart', 'I-MineralProperties', 'I-StateOfBeing', 'B-Sky', 'B-LivingDying', 'B-Evolution', 'B-Death', 'I-NorthernHemisphereLocations', 'I-PartsOfAVirus', 'B-Monera', 'B-AtomicProperties', 'B-Associate', 'B-PartsOfEndocrineSystem', 'B-Vacuum', 'I-Appliance', 'B-Pattern', 'B-Create', 'I-CleanUp', 'B-Day', 'B-Unit', 'B-UnderwaterEcosystem', 'I-Protist', 'I-ActionsForTides', 'I-Magnetic', 'I-MoneyTerms', 'I-Release', 'I-ResultsOfDecomposition', 'B-Wetness', 'B-Sedimentary', 'B-DigestiveSystem', 'I-Relations', 'I-Sky', 'B-SensoryTerms', 'B-PullingForces', 'I-RelativeDirection', 'B-Reproduction', 'B-LayersOfTheEarth', 'I-TypesOfTerrestrialEcosystems', 'I-PartsOfTheRespiratorySystem', 'B-ContainBeComposedOf', 'B-Relations', 'B-Compete', 'I-SeparatingMixtures', 'I-PERCENT', 'B-PlantNutrients', 'I-Medicine', 'I-Energy', 'B-Pressure', 'B-Constellation', 'B-CleanUp', 'B-DistanceMeasuringTools', 'I-FiltrationTool', 'I-Genetics', 'B-Exemplar', 'I-PlantNutrients', 'B-WordsRelatingToCosmologicalTheoriesExpandContract', 'B-LivingThing', 'B-VisualProperty', 'B-Representation', 'B-Senses', 'B-PhysicalProperty', 'I-Gene', 'I-Sedimentary', 'I-CelestialMeasurements', 'B-SeparatingMixtures', 'I-CelestialEvents', 'B-Particles', 'B-PostnatalOrganismStages', 'I-Rock', 'B-ChangesToResources', 'B-Meteorology', 'B-ExcretoryActions', 'I-AtmosphericLayers', 'I-Evolution', 'B-PushingActions', 'I-Problem', 'B-Harm', 'I-Nebula', 'I-PartsOfEndocrineSystem', 'I-ScientificAssociationsAdministrations', 'I-Cost', 'I-MedicalTerms', 'B-Hypothesizing', 'B-Succeed', 'I-LightProducingObject', 'I-ElectricalUnit', 'I-EarthPartsGrossGroundAtmosphere', 'B-Light', 'I-SeasonsFallAutumnWinterSpringSummer', 'B-PartsOfTheMuscularSystem', 'B-ScientificAssociationsAdministrations', 'I-CapillaryAction', 'B-ElectricalEnergy', 'B-ImmuneSystem', 'I-ConstructiveDestructiveForces', 'B-Metamorphic', 'I-Identify', 'B-NorthernHemisphereLocations', 'I-Goal', 'B-Position', 'B-PushingForces', 'B-ExamplesOfHabitats', 'B-ViewingTools', 'B-ObservationTechniques', 'B-LocationChangingActions', 'I-TIME', 'B-AmountChangingActions', 'B-AbsorbEnergy', 'I-GeneticRelations', 'B-ElectromagneticSpectrum', 'B-AnimalSystemsProcesses', 'B-OtherOrganismProperties', 'B-Validity', 'I-Associate', 'B-SpaceMissionsEGApolloGeminiMercury', 'I-BacteriaPart', 'I-Hypothesizing', 'B-MassUnit', 'I-EnvironmentalDamageDestruction', 'B-IncreaseDecrease', 'B-Flammability', 'I-ComputingDevice', 'B-Gene', 'I-ProbabilityAndCertainty', 'B-NervousSystem', 'I-SpaceProbes', 'I-PartsOfTheImmuneSystem', 'B-ResultsOfDecomposition', 'B-OtherAnimalProperties', 'B-PlantProcesses', 'B-Complexity', 'B-Start', 'B-SpacecraftSubsystem', 'B-Release', 'B-Growth', 'B-AbilityAvailability', 'B-NUMBER', 'I-PhaseChangingActions', 'B-PhysicalActivity', 'B-FossilForming', 'B-Amphibian', 'I-Shape', 'B-ApparentCelestialMovement', 'B-Matter', 'B-Toxins', 'I-SoundProducingObject', 'I-Unknown', 'B-LandVehicle', 'I-Habitat', 'I-Fossils', 'B-Currents', 'B-Value', 'I-TechnologicalInstrument', 'I-Taxonomy', 'B-DURATION', 'B-Surpass', 'B-LightProducingObject', 'I-PERSON', 'B-Transportation', 'I-GroupsOfOrganisms', 'B-DensityUnit', 'I-Observe', 'B-VolumeUnit', 'I-Development', 'I-Negations', 'B-MammalAnimalPart', 'I-TimesOfDayDayNight', 'I-Material', 'I-AirVehicle', 'B-TraceFossil', 'B-TypesOfIllness', 'I-PhysicalProperty', 'B-MarkersOfTime', 'B-CookingToolsFood', 'I-Products', 'I-ArithmeticMeasure', 'B-CardinalDirectionsNorthEastSouthWest', 'I-ClassesOfElements', 'I-GroupsOfScientists', 'B-OrganicCompounds', 'B-VehicularSystemsParts', 'B-TIME', 'I-Mass', 'I-AnimalPart', 'B-NonlivingPartsOfTheEnvironment', 'B-ElectricalUnit', 'B-RespirationActions', 'B-Brightness', 'I-EcosystemsEnvironment', 'B-Cell', 'I-PlanetParts', 'B-Comparisons', 'I-Reactions', 'I-EnvironmentalPhenomena', 'B-StateOfBeing', 'B-PercentUnit', 'I-Groups', 'B-ComputingDevice', 'B-PartsOfTheIntegumentarySystem', 'I-Cycles', 'B-BlackHole', 'B-Buy', 'I-ConstructionTools', 'I-VariablesControls', 'I-Homeostasis', 'B-Discovery', 'B-EnergyUnit', 'B-PartsOfTheNervousSystem', 'B-Meteor', 'B-ScientificMethod', 'I-PhaseTransitionPoint', 'I-PartsOfTheDigestiveSystem', 'I-ElectricalEnergy', 'I-PartsOfObservationInstruments', 'I-StarLayers', 'B-NaturalMaterial', 'I-ActionsForNutrition', 'I-NaturalResources', 'I-ElectromagneticSpectrum', 'B-Negations', 'B-ChemicalProperty', 'B-MineralFormations', 'B-Undiscovered', 'B-GeometricUnit', 'I-ExcretoryActions', 'I-PerformAnActivity', 'B-LOCATION', 'B-Soil', 'I-BlackHole', 'I-EmergencyServices', 'I-Extinction', 'B-QuestionActivityType', 'B-ExcretorySystem', 'B-PropertyOfProduction', 'B-Unknown', 'B-MedicalTerms', 'B-Rigidity', 'B-WordsForData', 'I-ElectricityGeneration', 'I-PullingActions', 'B-PhaseTransitionPoint', 'I-Geography', 'I-BusinessNames', 'B-ConservationLaws', 'I-CelestialObject', 'I-EnergyWaves', 'B-YearNumerals', 'I-Gymnosperm', 'B-PartsOfABusiness', 'I-PropertyOfMotion', 'B-EndocrineActions', 'B-CirculationActions', 'I-InsectAnimalPart', 'I-CellProcesses', 'B-Star', 'B-Gravity', 'I-Cause', 'B-HumanPart', 'B-LiquidHoldingContainersRecepticles', 'I-MineralFormations', 'B-Move', 'B-LightExaminingTool', 'B-PartsOfTheCirculatorySystem', 'B-Permit', 'B-TimesOfDayDayNight', 'I-SystemParts', 'B-AstronomyAeronautics', 'B-Birth', 'B-Homeostasis', 'I-Reptile', 'I-WordsForData', 'I-Igneous', 'B-Distance', 'I-ScientificMeetings', 'I-Discovery', 'B-GeneticProcesses', 'B-StateOfMatter', 'B-ElectricityMeasuringTool', 'B-PartsOfWaterCycle', 'I-AbilityAvailability', 'B-Force', 'I-AquaticAnimalPart', 'B-PullingActions', 'B-EmergencyServices', 'I-Senses', 'B-OutbreakClassification', 'I-TypeOfConsumer', 'I-Angiosperm', 'B-Biology', 'B-EnergyWaves', 'I-ChemicalProduct', 'I-DistanceMeasuringTools', 'I-MarkersOfTime', 'I-Cell', 'I-ImmuneSystem', 'I-Wetness', 'B-ChangeInComposition', 'I-LunarPhases', 'I-Fungi', 'B-Result', 'I-ActionsForAnimals', 'B-ImportanceComparison', 'I-TypesOfIllness', 'I-Divide', 'B-GenericTerms', 'B-SystemOfCommunication', 'B-BirdAnimalPart', 'I-Light', 'B-ClassesOfElements', 'B-FrequencyUnit', 'I-Permit', 'B-InheritedBehavior', 'B-SystemAndFunctions', 'B-ChangeInLocation', 'B-GuidelinesAndRules', 'B-Preserve', 'I-Toxins', 'I-ElectricalProperty', 'B-Vehicle', 'B-SyntheticMaterial', 'B-SafetyEquipment', 'B-PropertiesOfFood', 'I-SystemOfCommunication', 'I-PhaseChanges', 'B-Satellite', 'B-AmphibianAnimalPart', 'B-Break', 'B-Classify', 'B-PerformingResearch', 'I-ManMadeGeographicFormations', 'I-SimpleMachines', 'I-SolidMatter', 'I-Rigidity', 'B-GaseousMatter', 'B-Angiosperm', 'B-ExamplesOfSounds', 'I-Width', 'B-CirculatorySystem', 'B-PerformAnActivity', 'B-Audiences', 'B-PH', 'B-AnimalClassificationMethod', 'I-SafetyEquipment', 'B-TemporalProperty', 'B-Rarity', 'I-LifeCycle', 'I-ScientificTheoryExperimentationAndHistory', 'I-Forests', 'B-ObjectPart', 'B-MuscularSystem', 'B-TypesOfWaterInBodiesOfWater', 'B-RelativeTime', 'I-ImportanceComparison', 'I-AnalyzingResearch', 'B-NationalityOrigin', 'I-CastFossilMoldFossil', 'I-GranularSolids', 'I-IncreaseDecrease', 'I-Particles', 'B-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'I-PartsOfBodiesOfWater', 'B-Language', 'I-Viewpoint', 'B-AreaUnit', 'B-GeologicTheories', 'B-OtherDescriptionsForPlantsBiennialLeafyEtc', 'B-ReleaseEnergy', 'B-Width', 'B-ObservationPlacesEGObservatory', 'I-ResistanceStrength', 'B-Viewpoint', 'B-AtmosphericLayers', 'I-PropertiesOfSickness', 'B-CellsAndGenetics', 'B-BeliefKnowledge', 'B-PartsOfTheFoodChain', 'I-TransferEnergy', 'B-AcidityUnit', 'I-UnderwaterEcosystem', 'I-AbsorbEnergy', 'B-Speciation', 'I-ReleaseEnergy', 'I-SpeedUnit', 'I-FossilForming', 'B-Bird', 'B-PhaseChanges', 'B-DwarfPlanets', 'B-MoneyTerms', 'I-LandVehicle', 'I-NationalityOrigin', 'I-GeologicTheories', 'B-MagneticForce', 'B-TimeUnit', 'I-RespiratorySystem', 'B-ORGANIZATION', 'I-LivingThing', 'I-StopRemove', 'B-HeatingAppliance', 'B-ClothesTextiles', 'B-Circuits', 'I-GeographicFormationProcess', 'I-LayersOfTheEarth', 'I-NaturalPhenomena', 'I-ExamplesOfSounds', 'B-Age', 'B-Animal', 'B-PartsOfObservationInstruments', 'I-Amphibian', 'B-Asteroid', 'I-MassMeasuringTool', 'I-OtherHumanProperties', 'B-MeasuresOfAmountOfLight', 'I-CellsAndGenetics', 'B-Patents', 'B-Source', 'B-Reactions', 'I-PowerUnit', 'B-CoolingToolsFood', 'I-SouthernHemisphereLocations', 'B-GeometricMeasurements', 'B-GeographicFormationProcess', 'I-Traffic', 'I-PushingForces', 'B-Bryophyte', 'O', 'B-Locations', 'B-Texture', 'B-PropertiesOfWaves', 'B-Verify', 'B-SpeedUnit', 'B-Adaptation', 'I-PhasesOfWater', 'I-CirculationActions', 'I-HumanPart', 'B-Experimentation', 'I-Asteroid', 'B-Event', 'B-TerrestrialLocations', 'I-NaturalSelection', 'B-Alter', 'B-LightMovement', 'B-Comet', 'B-Nutrition', 'B-Changes', 'B-Foods', 'I-ParticleMovement', 'B-StarLayers', 'B-Inertia', 'I-TemperatureUnit', 'I-VerbsForLocate', 'B-OtherProperties', 'B-PartsOfChemicalReactions', 'B-Compound', 'B-Forests', 'I-PerformingExperimentsWell', 'I-States', 'B-TemperatureUnit', 'B-ElectricAppliance', 'B-SolarSystem', 'I-Countries', 'B-Examine', 'I-Temperature', 'I-MuscularSystemActions', 'B-ChemicalChange', 'B-MagneticEnergy', 'I-ElectricityAndCircuits', 'I-AnimalClassificationMethod', 'B-PartsOfABuilding', 'I-ElementalComponents', 'B-PartsOfTheImmuneSystem', 'B-CoolingAppliance', 'B-LearnedBehavior', 'I-Result', 'B-CelestialObject', 'I-SolarSystem', 'I-DistanceUnit', 'I-Aquatic', 'B-Nebula', 'I-WeightMeasuringTool', 'B-InsectAnimalPart', 'I-LivingDying', 'I-NonlivingPartsOfTheEnvironment', 'B-Precipitation', 'B-DATE', 'B-Human', 'I-Numbers', 'B-DistanceUnit', 'B-RespiratorySystem', 'B-PopularMedia', 'B-Touch', 'I-Color', 'I-SoundMeasuringTools', 'I-Uptake', 'I-IllnessPreventionCuring', 'B-MetalSolids', 'I-ChemicalChange', 'B-Agriculture', 'B-WordsForOffspring', 'B-GeneticRelations', 'B-DigestionActions', 'I-NervousSystem', 'B-AquaticAnimalPart', 'I-NutritiveSubstancesForAnimalsOrPlants', 'I-Believe', 'I-AnimalAdditionalCategories', 'I-WavePerception', 'I-CoolingAppliance', 'I-InheritedBehavior', 'I-Eukaryote', 'I-DigestiveSubstances', 'I-LiquidMovement', 'I-ArcheologicalProcessTechnique', 'B-EnvironmentalDamageDestruction', 'I-Speed', 'B-GeographicFormations', 'B-Fungi', 'B-Galaxy', 'B-ScientificMeetings', 'B-ConstructionTools', 'B-Bacteria', 'I-Behaviors', 'I-OtherOrganismProperties', 'B-Speed', 'B-Length', 'I-PartsOfABuilding', 'I-Reproduction', 'I-LocationChangingActions', 'B-SoundProducingObject', 'B-Health', 'I-Preserve', 'B-RepresentingElementsAndMolecules', 'I-DATE', 'B-Sickness', 'I-PhysicalActivity', 'I-TypesOfChemicalReactions', 'B-ColorChangingActions', 'B-Collect', 'I-PartsOfTheMuscularSystem', 'B-EnvironmentalPhenomena', 'I-Mutation', 'I-Compound', 'I-PartsOfTheNervousSystem', 'B-Cycles', 'B-Group', 'B-ActionsForNutrition', 'B-Believe', 'B-GeographicFormationParts', 'I-GalaxyParts', 'I-Collect', 'B-ChemicalProcesses', 'B-SystemProcessStages', 'B-Shape', 'B-EarthPartsGrossGroundAtmosphere', 'I-MagneticDevice', 'I-ObjectPart', 'B-Behaviors', 'I-ChemicalProcesses', 'B-NutritiveSubstancesForAnimalsOrPlants', 'B-ElectricalProperty', 'B-PressureUnit', 'I-SoundEnergy',
                'B-DigestiveSubstances', 'I-PartsOfTheReproductiveSystem', 'I-Human', "[CLS]", "[SEP]"]
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