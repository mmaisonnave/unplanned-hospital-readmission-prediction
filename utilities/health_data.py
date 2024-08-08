"""


"""
from dataclasses import dataclass, fields
from enum import Enum
import datetime
from typing import Union
from typing_extensions import Self
import numpy as np
import os
import gensim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from utilities import configuration
import json
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from utilities import data_manager
from utilities import io

class ReadmissionCode(Enum):
    """
    Enum representing different types of readmission codes.

    This enum categorizes readmission types into planned, unplanned within specific time frames, 
    new acute patients, other, and none. It also provides methods to check if a code is 
    associated with unplanned readmission and to get a string representation of the code.

    Attributes:
        PLANNED_READMIT (int): Code representing a planned readmission.
        UNPLANNED_READMIT_0_7 (int): Code representing an unplanned readmission within 7 days.
        UNPLANNED_READMIT_8_28 (int): Code representing an unplanned readmission between 8 and 28 days.
        UNPLANNED_FROM_SDS_0_7 (int): Code representing an unplanned readmission from SDS within 7 days.
        NEW_ACUTE_PATIENT (int): Code representing a new acute patient.
        OTHER (int): Code representing other types of readmissions.
        NONE (int): Code representing no readmission.
    """
    PLANNED_READMIT = 1
    UNPLANNED_READMIT_0_7 = 2
    UNPLANNED_READMIT_8_28 = 3
    UNPLANNED_FROM_SDS_0_7 = 4
    NEW_ACUTE_PATIENT = 5
    OTHER = 9
    NONE=-1

    @property
    def is_unplanned_readmit(self:Self) -> bool:
        """
        Check if the readmission code is one of the codes indicating an unplanned readmission.

        Returns:
            bool: True if the code indicates an unplanned readmission, otherwise False.
        """
        return self in (ReadmissionCode.UNPLANNED_READMIT_8_28,
                        ReadmissionCode.UNPLANNED_READMIT_0_7,
                        ReadmissionCode.UNPLANNED_FROM_SDS_0_7)
    @staticmethod
    def is_readmit(admission_code:Self) -> bool:
        """
        Check if the given admission code indicates any type of readmission.

        Args:
            admission_code (ReadmissionCode): The admission code to check.

        Returns:
            bool: True if the code indicates any type of readmission, otherwise False.
        """
        return admission_code in (ReadmissionCode.UNPLANNED_READMIT_8_28,
                                  ReadmissionCode.UNPLANNED_READMIT_0_7, 
                                  ReadmissionCode.PLANNED_READMIT,
                                  ReadmissionCode.UNPLANNED_FROM_SDS_0_7)
    
    def __str__(self: Self) -> str:   
        """
        Get the string representation of the readmission code.

        Returns:
            str: The name of the readmission code as a string.
        """     
        representation = ''
        if self == ReadmissionCode.PLANNED_READMIT:
            representation = 'Planned Readmit'
        elif self == ReadmissionCode.NEW_ACUTE_PATIENT:
            representation = 'New Acute Patient'
        elif self == ReadmissionCode.OTHER:
            representation = 'ReadmissionCode:Other'
        elif self == ReadmissionCode.NONE:
            representation = 'ReadmissionCode:None'
        elif self == ReadmissionCode.UNPLANNED_FROM_SDS_0_7:
            representation = 'Unplanned from SDS'
        elif self == ReadmissionCode.UNPLANNED_READMIT_0_7:
            representation = 'Unplanned within 7 days'
        elif self == ReadmissionCode.UNPLANNED_READMIT_8_28:
            representation = 'Unplanned 8 to 28 days'

        return representation

class ComorbidityLevel(Enum):
    """
    Enum representing different levels of comorbidity.

    This enum categorizes comorbidity levels into various stages, ranging from no comorbidity
    to higher levels of comorbidity. It also includes a "not applicable" and "none" category.

    Attributes:
        NO_COMORBIDITY (int): Code representing no comorbidity.
        LEVEL_1_COMORBIDITY (int): Code representing level 1 comorbidity.
        LEVEL_2_COMORBIDITY (int): Code representing level 2 comorbidity.
        LEVEL_3_COMORBIDITY (int): Code representing level 3 comorbidity.
        LEVEL_4_COMORBIDITY (int): Code representing level 4 comorbidity.
        NOT_APPLICABLE (int): Code representing a situation where comorbidity is not applicable.
        NONE (int): Code representing no specific comorbidity category.
    """
    NO_COMORBIDITY = 0
    LEVEL_1_COMORBIDITY = 1
    LEVEL_2_COMORBIDITY = 2
    LEVEL_3_COMORBIDITY = 3
    LEVEL_4_COMORBIDITY = 4
    NOT_APPLICABLE = 8
    NONE=-1

    def __str__(self: Self)->str:
        """
        Get the string representation of the comorbidity level.

        Returns:
            str: The name of the comorbidity level as a string.
        """
        representation = ''
        if self == ComorbidityLevel.NO_COMORBIDITY:
            representation = 'No Comorbidity'
        elif self == ComorbidityLevel.LEVEL_1_COMORBIDITY:
            representation = 'Level 1 Comorbidity'
        elif self == ComorbidityLevel.LEVEL_2_COMORBIDITY:
            representation = 'Level 2 Comorbidity'
        elif self == ComorbidityLevel.LEVEL_3_COMORBIDITY:
            representation = 'Level 3 Comorbidity'
        elif self == ComorbidityLevel.LEVEL_4_COMORBIDITY:
            representation = 'Level 4 Comorbidity'
        return representation


class TransfusionGiven(Enum):
    """
    Enum representing whether a transfusion was given.

    This enum indicates if a transfusion was administered or not. It includes categories for
    'Yes', 'No', and 'None', with the latter representing an undefined or non-applicable state.

    Attributes:
        NO (int): Code representing that no transfusion was given.
        YES (int): Code representing that a transfusion was given.
        NONE (int): Code representing a non-applicable or undefined state.
    """
    NO = 0
    YES = 1
    NONE=-1
    @property
    def received_transfusion(self: Self,)->bool:
        """
        Check if a transfusion was given.

        Returns:
            bool: True if the transfusion was given, otherwise False.
        """
        return self == TransfusionGiven.YES
    
    def __str__(self: Self) -> str:
        """
        Get the string representation of the transfusion status.

        Returns:
            str: The name of the transfusion status as a string.
        """
        representation=''
        if self == TransfusionGiven.YES:
            representation = 'Yes'
        elif self == TransfusionGiven.NO:
            representation = 'No'
        elif self == TransfusionGiven.NONE:
            representation = 'TransfusionGiven:None'
        return representation


class AdmitCategory(Enum):
    """
    Enum representing different categories of admission.

    This enum categorizes admissions into various types, including elective, newborn, cadaver,
    stillborn, and urgent. It also includes a 'none' category to represent a non-applicable or
    undefined state.

    Attributes:
        ELECTIVE (int): Code representing an elective admission.
        NEW_BORN (int): Code representing a newborn admission.
        CADAVER (int): Code representing a cadaver admission.
        STILLBORN (int): Code representing a stillborn admission.
        URGENT (int): Code representing an urgent admission.
        NONE (int): Code representing a non-applicable or undefined state.
    """
    ELECTIVE = 1
    NEW_BORN = 2
    CADAVER = 3
    STILLBORN = 5
    URGENT = 6 
    NONE = -1

    def __str__(self: Self) -> str:
        """
        Get the string representation of the admission category.

        Returns:
            str: The name of the admission category as a string.
        """
        representation=''
        if self == AdmitCategory.ELECTIVE:
            representation = 'Elective admit'
        elif self == AdmitCategory.NEW_BORN:
            representation = 'Newborn admit'
        elif self == AdmitCategory.CADAVER:
            representation = 'Cadaver admit'
        elif self == AdmitCategory.STILLBORN:
            representation = 'Stillborn admit'
        elif self == AdmitCategory.URGENT:
            representation = 'Urgent admit'
        elif self == AdmitCategory.NONE:
            representation = 'AdmitCategory:None'

        return representation


class Gender(Enum):
    """
    Enum representing different gender categories.

    This enum categorizes gender into several types, including male, female, undifferentiated,
    other, and none. It also provides properties to check if the gender is male or female.

    Attributes:
        FEMALE (int): Code representing the female gender.
        MALE (int): Code representing the male gender.
        UNDIFFERENTIATED (int): Code representing an undifferentiated or unspecified gender.
        OTHER (int): Code representing a gender other than male, female, or undifferentiated.
        NONE (int): Code representing a non-applicable or undefined gender.
    """
    FEMALE = 1
    MALE = 2
    UNDIFFERENTIATED = 3
    OTHER = 4
    NONE = -1

    @property
    def is_male(self:Self, )->bool:
        """
        Check if the gender is male.

        Returns:
            bool: True if the gender is male, otherwise False.
        """
        return self == Gender.MALE
    
    @property
    def is_female(self:Self, )->bool:
        """
        Check if the gender is female.

        Returns:
            bool: True if the gender is female, otherwise False.
        """
        return self == Gender.FEMALE
    
    def __str__(self: Self) -> str:
        """
        Get the string representation of the gender.

        Returns:
            str: The name of the gender as a string.
        """
        representation = ''
        if self == Gender.MALE:
            representation = 'Male'
        elif self == Gender.FEMALE:
            representation = 'Female'
        elif self == Gender.UNDIFFERENTIATED:
            representation = 'Gender:Undifferentiated'
        elif self == Gender.OTHER:
            representation = 'Gender:Other'
        elif self == Gender.NONE:
            representation = 'Gender:None'
        return representation


@dataclass
class Diagnosis:
    """
    A class representing a collection of diagnoses.

    This class stores lists of diagnosis codes, texts, and types. It provides a method to prepend 
    diagnoses from another `Diagnosis` instance, which adds the codes, texts, and types from the 
    provided instance to the beginning of the current instance's lists.

    Attributes:
        codes (List[str]): A list of diagnosis codes.
        texts (List[str]): A list of diagnosis texts.
        types (List[str]): A list of diagnosis types.

    Methods:
        prepend_diagnosis(diagnosis: 'Diagnosis') -> None:
            Prepend the diagnosis codes, texts, and types from another `Diagnosis` instance to the 
            current instance's lists.
    """
    codes: list[str]
    texts: list[str]
    types: list[str]    
    
    def prepend_diagnosis(self, diagnosis) -> None:
        """
        Prepend the diagnosis codes, texts, and types from another `Diagnosis` instance.

        This method updates the current instance's lists by adding the codes, texts, and types 
        from the provided `Diagnosis` instance to the beginning of the current lists.

        Args:
            diagnosis (Diagnosis): The `Diagnosis` instance whose codes, texts, and types 
                                   are to be prepended to the current instance.

        Returns:
            None: This method does not return a value.
        """
        self.codes = diagnosis.codes + self.codes
        self.texts = diagnosis.texts + self.texts
        self.types = diagnosis.types + self.types

class EntryCode(Enum):
    """
    Enum representing different types of entry codes.

    This enum categorizes various entry types into specific codes. It includes categories for clinic
    entry, direct entry, emergency entry, newborn entry, day surgery entry, stillborn entry, and
    a none category representing non-applicable or undefined entries.

    Attributes:
        NONE (int): Code representing no specific entry type.
        CLINIC_ENTRY (int): Code representing an clinic entry.
        DIRECT_ENTRY (int): Code representing a direct entry.
        EMERGENCY_ENTRY (int): Code representing an emergency entry.
        NEWBORN_ENTRY (int): Code representing an entry for a newborn.
        DAY_SURGERY_ENTRY (int): Code representing an entry for a day surgery.
        STILLBORN_ENTRY (int): Code representing an entry for a stillborn.

    Methods:
        __str__() -> str:
            Get the string representation of the entry code.
    """
    NONE=-1
    CLINIC_ENTRY = 1
    DIRECT_ENTRY = 2
    EMERGENCY_ENTRY = 3
    NEWBORN_ENTRY = 4
    DAY_SURGERY_ENTRY=5
    STILLBORN_ENTRY=6

    def __str__(self: Self) -> str:
        """
        Get the string representation of the entry code.

        Returns:
            str: The name of the entry code as a string.
        """
        representation = ''

        if self == EntryCode.NONE:
            representation = 'EntryCode:None'
        elif self == EntryCode.CLINIC_ENTRY:
            representation = 'Clinic Entry'
        elif self == EntryCode.DIRECT_ENTRY:
            representation = 'Direct Entry'
        elif self == EntryCode.EMERGENCY_ENTRY:
            representation = 'Emergency Entry'
        elif self == EntryCode.NEWBORN_ENTRY:
            representation = 'Newborn Entry'
        elif self == EntryCode.DAY_SURGERY_ENTRY:
            representation = 'Day Surgery Entry'
        elif self == EntryCode.STILLBORN_ENTRY:
            representation = 'Stillborn Entry'
        return representation


@dataclass
class Admission:
    """Represents a patient admission record in a healthcare system.

    Attributes:
        admit_id (int): Unique identifier for the admission.
        code (Union[int, None]): Encoded Health Card Number.
        institution_number (int): Identifier for the institution where the admission took place.
        admit_date (Union[datetime.datetime, None]): Date when the patient was admited.
        discharge_date (datetime.datetime): Date when the patient was discharged.
        readmission_code (ReadmissionCode): Code indicating the type of readmission.
        age (int): Age of the patient at the time of admission.
        gender (Gender): Gender of the patient.
        mrdx (str): Main diagnosis code.
        postal_code (str): Patient's postal code.
        diagnosis (Diagnosis): Detailed diagnosis information.
        intervention_code (list): List of intervention codes performed during the admission.
        px_long_text (list): Long text descriptions of procedures performed.
        admit_category (AdmitCategory): Category of the admission (e.g., elective, urgent).
        transfusion_given (TransfusionGiven): Indicates whether a transfusion was given.
        main_pt_service (str): Main patient service.
        cmg (Union[float, None]): Case Mix Group value.
        comorbidity_level (ComorbidityLevel): Level of comorbidity.
        case_weight (Union[float, None]): Weight assigned to the case.
        alc_days (int): Number of days classified as Alternate Level of Care (ALC).
        acute_days (int): Number of days classified as acute care.
        institution_to (str): Destination institution post-discharge.
        institution_from (str): Origin institution where admission occurred.
        institution_type (str): Type of institution.
        discharge_unit (str): 
        is_central_zone (bool): Flag indicating if the institution is in the central zone.
        entry_code (EntryCode): Code representing the entry type (e.g., emergency, direct).
        readmission (Self): The readmission record, if any.

    Methods:
        def has_missing(self:Self,)->bool
        def __iter__(self: Self)
        def __post_init__(self)
        def from_dict_data(admit_id:int, admission:dict) -> Self
        def __repr__(self: Self,)->str
        def diagnosis_codes_features(admissions: list[Self], 
                                     min_df, 
                                     vocabulary=None, 
                                     use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix)
        def intervention_codes_features(admissions: list[Self], 
                                        min_df, 
                                        vocabulary=None, 
                                        use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix)
        def diagnosis_embeddings(admissions: list[Self],
                                 model_name:str,
                                 use_cached:bool =True) -> pd.DataFrame
        def intervention_embeddings(admissions: list[Self],
                                    model_name:str,
                                    use_cached:bool=True,) -> pd.DataFrame: 
        def categorical_features(admissions: list[Self],main_pt_services_list=None) -> pd.DataFrame
        def is_valid_training_instance(self:Self)->bool
        def is_valid_testing_instance(self:Self)->bool
        def numerical_features(admissions: list[Self],) -> pd.DataFrame
        def get_diagnoses_mapping()
        def get_intervention_mapping()
        def get_y(admissions: list[Self])->np.ndarray
        def has_readmission(self: Self,)->bool
        def length_of_stay(self: Self)->int
        def is_valid_readmission(self, readmission: Self)->bool
        def add_readmission(self, readmission: Self)
        def get_training_testing_data(filtering=True, 
                                      combining_diagnoses=False,
                                      combining_interventions=False) -> list[Self]:
        def get_train_test_matrices(params)
        def fix_missings(self: Self, training: list[Self])
        def get_train_test_data(filtering=True,
                                combining_diagnoses=False,
                                combining_interventions=False) -> list[Self]:

        def get_heldout_data(filtering=True,
                             combining_diagnoses=False, 
                             combining_interventions=False) -> list[Self]:
        def get_both_train_test_matrices(params)
        def get_development_and_held_out_matrices(params)
    """    
    admit_id: int
    code: Union[int,None]
    institution_number: int
    admit_date: Union[datetime.datetime,None]
    discharge_date: datetime.datetime
    readmission_code: ReadmissionCode
    age: int
    gender: Gender
    mrdx: str
    postal_code: str
    diagnosis: Diagnosis
    intervention_code:list
    px_long_text:list
    admit_category: AdmitCategory
    transfusion_given: TransfusionGiven
    main_pt_service:str
    cmg: Union[float,None]
    comorbidity_level:ComorbidityLevel
    case_weight: Union[float,None]
    alc_days: int
    acute_days: int
    institution_to: str
    institution_from: str
    institution_type: str
    discharge_unit:str
    is_central_zone: bool
    entry_code:EntryCode
    readmission: Self

    @property
    def has_missing(self:Self,)->bool:
        """
        Determines if any critical attributes of the admission record are missing or undefined.

        Specifically, it checks if any of the following 
        attributes are missing or set to default values indicating a lack of information:

        Attributes Checked `is None`:
            - `code`: Encoded Health Card Number. If missing, indicates the patient's HCN is not available.
            - `admit_date`: The date of admission. A None value indicates the admission date is unknown.
            - `gender`: The gender of the patient, where `Gender.NONE` indicates the gender is unspecified.
            - `main_pt_service`: The main patient service during admission. A None value indicates this information is missing.
            - `mrdx`: The main diagnosis code. A None value indicates no primary diagnosis has been recorded.
            - `entry_code`: The entry type of the admission, with `EntryCode.NONE` indicating it is undefined.

        Checked for NaN and is None:
        - `case_weight`: The weight assigned to the case, with None or NaN indicating it's missing.
        - `cmg`: Case Mix Group value. A None value or NaN indicates missing or undefined case weight.

        Enums checked for NONE:
            - `transfusion_given`: Indicates whether a transfusion was given, with `TransfusionGiven.NONE` signaling an undefined state.
            - `readmission_code`: The code indicating the type of readmission. A value of `ReadmissionCode.NONE` indicates no valid readmission code.
            - `admit_category`: The category of the admission. A value of `AdmitCategory.NONE` means the admission type is not specified.

        Returns:
            bool: Returns `True` if any of the above attributes are missing or undefined, otherwise returns `False`.
        """
        return self.code is None or \
               self.cmg is None or \
               np.isnan(self.cmg) or \
               self.case_weight is None or \
               np.isnan(self.case_weight) or \
               self.admit_date is None or \
               self.readmission_code == ReadmissionCode.NONE or \
               self.gender == Gender.NONE or \
               self.admit_category == AdmitCategory.NONE or \
               self.main_pt_service is None or \
               self.mrdx is None or \
               self.entry_code == EntryCode.NONE or \
               self.transfusion_given == TransfusionGiven.NONE

    def __iter__(self: Self):
        """
        Allows iteration over the attributes of the `Admission` instance (getattr) as key-value pairs.

        This method makes the `Admission` class iterable, enabling it to be used in loops or other 
        contexts where iteration over its attributes is required. Each iteration yields a tuple 
        containing the attribute name and its corresponding value.

        Returns:
            generator: A generator that produces tuples of the form (`attribute_name`, `attribute_value`) 
            for each attribute defined in the `Admission` class.
        
        Example:
            >>> admission = Admission(admit_id=1, ...)
            >>> for attr, value in admission:
            >>>     print(f"{attr}: {value}")
        """
        return ((field.name, getattr(self, field.name)) for field in fields(self))

    def __post_init__(self):
        """
        Performs additional validation and initialization after the `Admission` instance is created.

        This method is automatically called after the `Admission` instance is initialized, ensuring 
        that the data provided is consistent and logically sound. It performs the following checks:

        - Ensures that the `admit_date` is not later than the `discharge_date`.
        - Validates that the `age` is non-negative for all patients except for those in the 
        `AdmitCategory.NEW_BORN`.

        Raises:
            AssertionError: If the `admit_date` is later than the `discharge_date`.
            AssertionError: If the `age` is negative for non-newborns, or if the `age` is outside 
            the valid range for newborns.
        
        Example:
            >>> admission = Admission(admit_date=datetime(2024, 8, 1), discharge_date=datetime(2024, 8, 3), ...)
            >>> # This will raise an assertion error if the conditions are not met.
        """ 
        if not self.admit_date is None:
            assert self.admit_date <= self.discharge_date

        if self.admit_category != AdmitCategory.NEW_BORN:
            assert 0<=self.age
        else: # NEW BORN
            assert -1<=self.age

    @staticmethod
    def from_dict_data(admit_id:int, admission:dict) -> Self:
        """
        Creates and returns a new `Admission` instance from a dictionary containing admission data.

        This method interprets and converts the data stored in a dictionary into an `Admission` object. 
        It maps the dictionary keys to the corresponding attributes in the `Admission` class and handles 
        the conversion of various attributes, including enumerations, dates, and numerical values.

        Args:
            admit_id (int): The unique identifier for the admission.
            admission (dict): A dictionary containing the admission data, where keys correspond to 
                            attribute names and values hold the data to be assigned to those attributes.

        Returns:
            Self: An `Admission` instance populated with data from the provided dictionary.

        Raises:
            AssertionError: If the input data for specific attributes does not match the expected values 
                        or is inconsistent (e.g., invalid gender or entry code).

        Example:
            >>> data = {
            >>>     "HCN code": 123456789,
            >>>     "Institution Number": 456,
            >>>     "Admit Date": "2024-08-01",
            >>>     "Discharge Date": "2024-08-10",
            >>>     ...
            >>> }
            >>> admission = Admission.from_dict_data(admit_id=1, admission=data)
        """
        # Readmission code
        readmission_code = ReadmissionCode(int(admission['Readmission Code'][0])) if not admission['Readmission Code'] is None else ReadmissionCode.NONE

        #Diagnosis 
        diagnosis = Diagnosis(codes=admission['Diagnosis Code'], texts=admission['Diagnosis Long Text'] , types=admission['Diagnosis Type'])

        # Admit Category
        if admission['Admit Category'] is None:
            admit_category = AdmitCategory.NONE
        elif 'Elective' in admission['Admit Category']:
            admit_category = AdmitCategory.ELECTIVE
        elif 'Newborn' in admission['Admit Category']:
            admit_category = AdmitCategory.NEW_BORN
        elif 'Cadaver' in admission['Admit Category']:
            admit_category = AdmitCategory.CADAVER
        elif 'Stillborn' in admission['Admit Category']:
            admit_category = AdmitCategory.STILLBORN
        else:
            assert 'urgent' in admission['Admit Category']
            admit_category = AdmitCategory.URGENT

        if admission['Transfusion Given'] is None:
            transfusion = TransfusionGiven.NONE
        elif admission['Transfusion Given']=='Yes':
            transfusion = TransfusionGiven.YES
        else:
            assert admission['Transfusion Given']=='No'
            transfusion = TransfusionGiven.NO

        if admission['Gender']=='Male':
            gender=Gender.MALE
        elif admission['Gender']=='Female':
            gender=Gender.FEMALE
        elif admission['Gender']=='Other (transsexu':
            gender=Gender.OTHER
        elif admission['Gender']=='Undifferentiated':
            gender=Gender.UNDIFFERENTIATED
        else: 
            assert admission['Gender'] is None
            gender=Gender.NONE

        if admission['Entry Code'] is None:
            entry_code = EntryCode.NONE
        elif admission['Entry Code']=='C Clinic from report':
            entry_code = EntryCode.CLINIC_ENTRY
        elif admission['Entry Code']=='D Direct':
            entry_code = EntryCode.DIRECT_ENTRY
        elif admission['Entry Code']=='E Emergency Departme':
            entry_code = EntryCode.EMERGENCY_ENTRY
        elif admission['Entry Code']=='N Newborn':
            entry_code = EntryCode.NEWBORN_ENTRY
        elif admission['Entry Code']=='S Stillborn':
            entry_code = EntryCode.STILLBORN_ENTRY
        else:
            assert admission['Entry Code']=='P Day Surgery from r', f"Invalid entry code found: {admission['Entry Code']}."
            entry_code = EntryCode.DAY_SURGERY_ENTRY

        # Readmission code
        comorbidity_level = ComorbidityLevel(int(admission['Comorbidity Level'][0])) if not admission['Comorbidity Level'] is None else ComorbidityLevel.NONE
        admission = Admission(admit_id=int(admit_id),
                        code=int(admission['HCN code']) if not admission['HCN code']is None else None,
                        institution_number = int(admission['Institution Number']),
                        admit_date = datetime.datetime.fromisoformat(admission['Admit Date']) if not admission['Admit Date'] is None else None,
                        discharge_date = datetime.datetime.fromisoformat(admission['Discharge Date']),
                        readmission_code = readmission_code,
                        age = int(admission['Patient Age']),
                        gender = gender,
                        mrdx = str(admission['MRDx']),
                        postal_code = str(admission['Postal Code']),
                        diagnosis = diagnosis,
                        intervention_code = admission['Intervention Code'],
                        px_long_text = admission['Px Long Text'],
                        admit_category = admit_category,
                        transfusion_given = transfusion,
                        main_pt_service = admission['Main Pt Service'],
                        cmg = float(admission['CMG']) if not admission['CMG'] is None else None,
                        comorbidity_level = comorbidity_level,
                        case_weight = float(admission['Case Weight']) if not admission['Case Weight'] is None else None,
                        alc_days = int(admission['ALC Days']),
                        acute_days = int(admission['Acute Days']),
                        institution_to = admission['Institution To'],
                        institution_from = admission['Institution From'],
                        institution_type = admission['Institution Type'],
                        discharge_unit = admission['Discharge Nurse Unit'],
                        is_central_zone = admission['CZ Status']=='cz',
                        entry_code = entry_code,
                        readmission=None
                        )
        return admission
    

    def __repr__(self: Self,)->str:
        """
        Provides a string representation of the `Admission` instance.

        This method returns a human-readable string that succinctly represents the 
        key attributes of an `Admission` object, making it useful for debugging 
        and logging purposes. The string includes information such as the patient's 
        code, admission and discharge dates, age, gender, number of ALC and acute days, 
        and whether the patient was readmitted.

        If the patient was readmitted, the string representation will include details 
        about the readmission, such as the readmission date, discharge date, and 
        readmission code.

        Returns:
            str: A string representation of the `Admission` instance.

        Example:
            >>> admission = Admission(...)
            >>> print(repr(admission))
            <Admission Patient_code='123456789' admit='2024-08-01' discharged='2024-08-10' Age='45' gender='MALE' ALC_days='2' acute_days='5' readmited=No>
            >>> # If readmitted
            <Admission Patient_code='123456789' admit='2024-08-01' discharged='2024-08-10' Age='45' gender='MALE' ALC_days='2' 
                                                acute_days='5' readmited(2024-09-01,2024-09-05,ReadmissionCode.UNPLANNED_READMIT_0_7)>
        """
        repr_ = f"<Admission Patient_code='{self.code}' "\
            f"admit='{self.admit_date.date()}' "\
                f"discharged='{self.discharge_date.date()}' "\
                    f"Age='{self.age}' gender='{self.gender}' ALC_days='{self.alc_days}' acute_days='{self.acute_days}' readmited=No>"
        if not self.readmission is None:
            repr_ = repr_[:-13] + f'readmited({self.readmission.admit_date.date()},{self.readmission.discharge_date.date()},{self.readmission.readmission_code})>'
        return repr_
    
    @staticmethod
    def diagnosis_codes_features(admissions: list[Self], min_df, vocabulary=None, use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix):
        """
        Computes a  matrix representing diagnosis codes from a list of `Admission` instances.

        Resulting matrix is binary if use_idf=False.

        This method takes a list of `Admission` objects and processes the diagnosis codes to generate 
        a matrix that can be used for machine learning tasks. It uses the `TfidfVectorizer` to convert 
        diagnosis codes into a sparse matrix, where each row corresponds to an admission and each column 
        corresponds to a diagnosis code feature.

        Args:
            admissions (list[Self]): A list of `Admission` instances from which to extract diagnosis codes.
            min_df (int or float): The minimum number (or proportion) of documents a term must appear in to be included in the vocabulary. 
            vocabulary (dict or list, optional): A predefined vocabulary for the vectorizer. If None, the vocabulary is built from the admissions data. Defaults to None.
            use_idf (bool, optional): Whether to enable inverse-document-frequency reweighting. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: An array of feature names corresponding to the diagnosis codes.
                - sparse._csr.csr_matrix: A sparse matrix where rows represent admissions and columns represent diagnosis code features.

        Example:
            >>> admissions = [admission1, admission2, admission3]
            >>> features, matrix = Admission.diagnosis_codes_features(admissions, min_df=1)
            >>> print(features)  # Array of feature names
            >>> print(matrix)  # Sparse matrix of diagnosis codes
        """  
        codes = [' '.join(admission.diagnosis.codes) for admission in admissions]
        if vocabulary is None:
            vectorizer = TfidfVectorizer(use_idf=use_idf,
                                         min_df=min_df,
                                         binary=True,
                                         norm=None,
                                        ).fit(codes)
        else:
            vectorizer = TfidfVectorizer(use_idf=use_idf, 
                                         min_df=min_df,
                                         binary=True,
                                         norm=None,
                                         vocabulary=vocabulary).fit(codes)

        return vectorizer.get_feature_names_out(), vectorizer.transform(codes)
    
    @staticmethod
    def intervention_codes_features(admissions: list[Self], min_df, vocabulary=None, use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix):
        """
        Computes a matrix representing intervention codes from a list of `Admission` instances.

        Resulting matrix is binary if use_idf=False.

        This method processes the intervention codes from a list of `Admission` objects to generate 
        a matrix suitable for machine learning tasks. It uses the `TfidfVectorizer` to convert 
        intervention codes into a sparse matrix, where each row represents an admission, and each column 
        represents an intervention code feature.

        Args:
            admissions (list[Self]): A list of `Admission` instances from which to extract intervention codes.
            min_df (int or float): The minimum number (or proportion) of documents a term must appear in to be included in the vocabulary. 
            vocabulary (dict or list, optional): A predefined vocabulary for the vectorizer. If None, the vocabulary is built from the admissions data. Defaults to None.
            use_idf (bool, optional): Whether to enable inverse-document-frequency reweighting. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: An array of feature names corresponding to the intervention codes.
                - sparse._csr.csr_matrix: A sparse matrix where rows represent admissions and columns represent intervention code features.

        Example:
            >>> admissions = [admission1, admission2, admission3]
            >>> features, matrix = Admission.intervention_codes_features(admissions, min_df=1)
            >>> print(features)  # Array of feature names
            >>> print(matrix)  # Sparse matrix of intervention codes
        """
        codes = [' '.join(admission.intervention_code) for admission in admissions]
        if vocabulary is None:
            vectorizer = TfidfVectorizer(use_idf=use_idf,
                                         min_df=min_df,
                                         binary=True,
                                         norm=None,
                                        ).fit(codes)
        else:
            vectorizer = TfidfVectorizer(use_idf=use_idf, 
                                         min_df=min_df,
                                         binary=True,
                                         norm=None,
                                         vocabulary=vocabulary).fit(codes)
        return vectorizer.get_feature_names_out(), vectorizer.transform(codes)

    @staticmethod
    def diagnosis_embeddings(admissions: list[Self],
                             model_name:str,
                             use_cached:bool =True) -> pd.DataFrame:
        """
        Generates a DataFrame of diagnosis embeddings for a list of `Admission` instances using a pre-trained Gensim model.
        
        This method processes a list of `Admission` objects and converts their diagnosis codes into vector embeddings 
        using a pre-trained Gensim model specified by `model_name`. The embeddings are stored in a DataFrame, where 
        each row corresponds to an admission, and each column represents a dimension of the embedding space.

        If a cached embedding matrix is available and `use_cached` is set to `True`, the method will load the embeddings 
        from disk. Otherwise, it will compute the embeddings from scratch using the Gensim model.

        The method first checks if a cached embedding matrix exists on disk (embedding_full_path). If the cached 
        embeddings are available and use_cached is set to True, the method loads these embeddings directly, bypassing 
        the need for computation. If the cached embeddings are not used or don't exist, the method computes the embeddings 
        from scratch by loading the specified Gensim model and applying it to the diagnosis codes.

        For any admissions without precomputed embeddings, the method infers their embeddings using the pre-trained Gensim 
        model. 
        
        The method will fail if neither cached embeddings nor the pre-trained model are available.


        Args:
            admissions (list[Self]): A list of `Admission` instances for which to generate diagnosis embeddings.
            model_name (str): The name of the pre-trained Gensim model to use for generating the embeddings.
            use_cached (bool, optional): If `True`, use a cached version of the embeddings if available. 
                                        If `False`, compute embeddings from scratch and optionally save them. 
                                        Defaults to `True`.

        Returns:
            pd.DataFrame: A DataFrame containing the diagnosis embeddings, with as many rows as there are 
                        admissions in the input list, and as many columns as the number of dimensions 
                        in the embedding model.

        Example:
            >>> admissions = [admission1, admission2, admission3]
            >>> embeddings_df = Admission.diagnosis_embeddings(admissions, model_name='diagnosis_model')
            >>> print(embeddings_df)  # DataFrame of diagnosis embeddings
        """
        print('Computing diagnosis embeddings ...')
        config = configuration.get_config()
        full_model_path = os.path.join(config['gensim_model_folder'], model_name)
        embedding_full_path = full_model_path+'_embeddings.npy'

        admission2embedding = {}
        # precomputed_found = 0
        if os.path.isfile(embedding_full_path) and not use_cached:
            print('Precomputed embeddings found but NOT BEING USED (diagnosis)')
        if os.path.isfile(embedding_full_path) and use_cached:
            matrix = np.load(embedding_full_path)
            # If embedding_size=100, with Y number of admissions the matrix is (101, Y) shaped
            # The first column is the admit_id, the other 100 dimensions are the embeddings.
            admission2embedding = {admit_id: matrix[ix,1:] 
                                   for ix, admit_id in enumerate(matrix[:,0])}
            # precomputed_found = len(admission2embedding)   
        else:
            print('NOT USING CACHED EMBEDDINGS (diagnosis)')
        remaining_to_compute = [admission
                                for admission in admissions 
                                if not admission.admit_id in admission2embedding]

        if len(remaining_to_compute)!=0:
            model = gensim.models.doc2vec.Doc2Vec.load(full_model_path)
            admission2embedding |= {admission.admit_id: model.infer_vector(admission.diagnosis.codes)
                                for admission in remaining_to_compute}
            admit_ids = np.array([admit_id for admit_id in admission2embedding])
            matrix = np.vstack([admission2embedding[admit_id] for admit_id in admit_ids])
            to_store = np.hstack([admit_ids.reshape(-1,1),matrix])
            if use_cached:
                np.save(embedding_full_path, to_store)
            else:
                print('NOT SAVING, BECAUSE use_cached==False')

        # Get embedding size from first admission
        emb_size = admission2embedding[admissions[0].admit_id].shape[0]

        return pd.DataFrame(np.vstack([admission2embedding[admission.admit_id] for admission in admissions]),
                            columns=[f'emb_{ix}' for ix in range(emb_size)]
                            )


    @staticmethod
    def intervention_embeddings(admissions: list[Self], 
                                model_name:str,
                                use_cached:bool=True,
                                ) -> pd.DataFrame:
        """
        Generates a DataFrame of intervention embeddings for a list of `Admission` instances using a pre-trained Gensim model.

        This method processes a list of `Admission` objects and converts their intervention codes into vector embeddings 
        using a pre-trained Gensim model specified by `model_name`. The embeddings are stored in a DataFrame, where each row 
        corresponds to an admission, and each column represents a dimension of the embedding space.

        If a cached embedding matrix is available and `use_cached` is set to `True`, the method will load the embeddings 
        from disk. Otherwise, it will compute the embeddings from scratch using the pre-trained Gensim model.

        The method first checks if a cached embedding matrix exists on disk (`embedding_full_path`). If the cached embeddings 
        are available and `use_cached` is set to `True`, the method loads these embeddings directly, bypassing the need for 
        computation. If the cached embeddings are not used or don't exist, the method computes the embeddings from scratch by 
        loading the specified Gensim model and applying it to the intervention codes.

        For any admissions without precomputed embeddings, the method infers their embeddings using the pre-trained Gensim 
        model.

        The method will fail if neither cached embeddings nor the pre-trained model are available.



        Args:
            admissions (list[Self]): A list of `Admission` instances for which to generate intervention embeddings.
            model_name (str): The name of the pre-trained Gensim model to use for generating the embeddings.
            use_cached (bool, optional): If `True`, use a cached version of the embeddings if available. 
                                        If `False`, compute embeddings from scratch and optionally save them. 
                                        Defaults to `True`.

        Returns:
            pd.DataFrame: A DataFrame containing the intervention embeddings, with as many rows as there are admissions in 
                        the input list, and as many columns as the number of dimensions in the embedding model.

        Example:
            >>> admissions = [admission1, admission2, admission3]
            >>> embeddings_df = Admission.intervention_embeddings(admissions, model_name='intervention_model')
            >>> print(embeddings_df)  # DataFrame of intervention embeddings
        """
        config = configuration.get_config()
        full_model_path = os.path.join(config['gensim_model_folder'], model_name)
        embedding_full_path = full_model_path+'_embeddings.npy'

        admission2embedding = {}
        precomputed_found=0

        if os.path.isfile(embedding_full_path) and not use_cached:
            print('Precomputed intervention model found but NOT BEING USED')
        if os.path.isfile(embedding_full_path) and use_cached:
            matrix = np.load(embedding_full_path)
            # If embedding_size=100, with Y number of admissions the matrix is (101, Y) shaped
            # The first column is the admit_id, the other 100 dimensions are the embeddings.
            admission2embedding = {admit_id: matrix[ix,1:] 
                                   for ix, admit_id in enumerate(matrix[:,0])}

            precomputed_found = len(admission2embedding)
            print(f'Precomputed embeddings found {precomputed_found}')
        else:
            print('NOT USING CACHED MODEL (DIAGNOSIS)')

        remaining_to_compute = [admission
                                for admission in admissions
                                if not admission.admit_id in admission2embedding]
        
        model = gensim.models.doc2vec.Doc2Vec.load(full_model_path)
        admission2embedding |= {admission.admit_id: model.infer_vector(admission.intervention_code)
                             for admission in remaining_to_compute}
        
        if precomputed_found==len(admission2embedding):
            print('No new embeddings were added. ')
        else:
            print(f'After adding more embeddings new size= {len(admission2embedding)}')
            print('Saving new embeddings to disk ...')
            admit_ids = np.array([admit_id for admit_id in admission2embedding])
            matrix = np.vstack([admission2embedding[admit_id] for admit_id in admit_ids])
            to_store = np.hstack([admit_ids.reshape(-1,1),matrix])
            if use_cached:
                # If combined embeddings are used, we don't store the precomputed embeddings
                np.save(embedding_full_path, to_store)
            else:
                print('NOT SAVING, BECAUSE use_cached==False')


        # Get embedding size from first admission
        emb_size = admission2embedding[admissions[0].admit_id].shape[0]

        return pd.DataFrame(np.vstack([admission2embedding[admission.admit_id] for admission in admissions]), 
                            columns=[f'emb_{ix}' for ix in range(emb_size)]
                            )
    
    
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # CATEGORICAL FEATURES
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    @staticmethod
    def categorical_features(admissions: list[Self],main_pt_services_list=None) -> pd.DataFrame:
        """
        Generates a DataFrame of categorical features for a list of `Admission` instances.

        This method creates a DataFrame where each row represents an admission and each column represents a categorical feature. 
        The features include demographic information, admission types, comorbidity levels, entry codes, readmission codes, and 
        indicators related to the COVID-19 pandemic. Additionally, the method includes features for primary services based on the 
        `main_pt_services_list`.

        If `main_pt_services_list` is not provided, the method will extract unique primary services from the `Admission` instances. 
        The method will then create binary features for each unique primary service.

        The resulting DataFrame has the following columns:
            - Gender: `male`, `female`
            - Transfusion status: `transfusion given`
            - Alternate Level of Care (ALC): `is alc`
            - Central Zone: `is central zone`
            - Admission Category: `elective admission`, `urgent admission`
            - Comorbidity levels: `level 1 comorbidity` through `level 4 comorbidity`
            - Entry code: `Clinic Entry`, `Direct Entry`, `Emergency Entry`, `Day Surgery Entry`
            - Readmission code: `New Acute Patient`, `Planned Readmit`, `Unplanned Readmit`
            - COVID-19 pandemic indicator: `COVID Pandemic`
            - Primary services: Features for each unique primary service from `main_pt_services_list`

        Args:
            admissions (list[Self]): A list of `Admission` instances for which to generate categorical features.
            main_pt_services_list (list[str], optional): A list of primary services to include as features. If `None`, the 
                                                        list is derived from the `Admission` instances. Defaults to `None`.

        Returns:
            pd.DataFrame: A DataFrame containing the categorical features with rows corresponding to admissions and columns 
                        corresponding to features. The DataFrame also includes the `main_pt_services_list`.

        Example:
            >>> admissions = [admission1, admission2, admission3]
            >>> features_df, services_list = Admission.categorical_features(admissions)
            >>> print(features_df)  # DataFrame of categorical features
        """    
        columns = ['male',
                   'female', 
                   'transfusion given', 
                   'is alc',
                   'is central zone',
                   'elective admission',
                   'urgent admission',
                   'level 1 comorbidity',
                   'level 2 comorbidity',
                   'level 3 comorbidity',
                   'level 4 comorbidity',
                   'Clinic Entry',
                   'Direct Entry',
                   'Emergency Entry',
                   'Day Surgery Entry',
                   'New Acute Patient',
                   'Panned Readmit',
                   'Unplanned Readmit',
                   'COVID Pandemic',
                   ]
        if main_pt_services_list is None:
            main_pt_services_list = list(set([admission.main_pt_service for admission in admissions]))

        main_pt_services_list = sorted(main_pt_services_list)

        service2idx = dict([(service,ix+len(columns)) for ix,service in enumerate(main_pt_services_list)])
        columns = columns + main_pt_services_list

        vectors = []
        for admission in admissions:
            vector = [1 if admission.gender.is_male else 0,
                     1 if admission.gender.is_female else 0,
                     1 if admission.transfusion_given.received_transfusion else 0,
                     1 if admission.alc_days > 0 else 0,
                     1 if admission.is_central_zone else 0,
                     1 if admission.admit_category==AdmitCategory.ELECTIVE else 0,
                    #  1 if admission.admit_category==AdmitCategory.NEW_BORN else 0,
                     1 if admission.admit_category==AdmitCategory.URGENT else 0,
                    #  1 if admission.comorbidity_level==ComorbidityLevel.NO_COMORBIDITY else 0,      # 8
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_1_COMORBIDITY else 0, # 7
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_2_COMORBIDITY else 0, # 8
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_3_COMORBIDITY else 0, # 9
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_4_COMORBIDITY else 0, # 10
                     1 if admission.entry_code==EntryCode.CLINIC_ENTRY else 0,
                     1 if admission.entry_code==EntryCode.DIRECT_ENTRY else 0,
                     1 if admission.entry_code==EntryCode.EMERGENCY_ENTRY else 0,
                     1 if admission.entry_code==EntryCode.DAY_SURGERY_ENTRY else 0,
                    #  1 if admission.entry_code==EntryCode.NEWBORN_ENTRY else 0,
                     1 if admission.readmission_code==ReadmissionCode.NEW_ACUTE_PATIENT else 0,
                     1 if admission.readmission_code==ReadmissionCode.PLANNED_READMIT else 0,
                     1 if admission.readmission_code.is_unplanned_readmit else 0,
                     1 if admission.discharge_date >= datetime.datetime(2020,3,5) else 0 # First case of community transmission
                    #  1 if admission.readmission_code.OTHER else 0,

                    ]
            if admission.comorbidity_level==ComorbidityLevel.LEVEL_2_COMORBIDITY:
                assert vector[7]==0 and vector[8] == 1 and vector[9] == 0 and vector[10] == 0
                vector[7]=1
            if admission.comorbidity_level==ComorbidityLevel.LEVEL_3_COMORBIDITY:
                assert vector[7]==0 and vector[8] == 0 and vector[9] == 1 and vector[10] == 0
                vector[7]=1
                vector[8]=1
            if admission.comorbidity_level==ComorbidityLevel.LEVEL_4_COMORBIDITY:
                assert vector[7]==0 and vector[8] == 0 and vector[9] == 0 and vector[10] == 1
                vector[7]=1
                vector[8]=1
                vector[9]=1
            vector = vector + [0]*len(main_pt_services_list)

            if admission.main_pt_service in service2idx:
                vector[service2idx[admission.main_pt_service]]=1
            vectors.append(vector)

        return pd.DataFrame(vectors, columns=columns), main_pt_services_list

    @property
    def is_valid_training_instance(self:Self)->bool:
        """
        Determines if the current instance is valid for training purposes.

        An instance is considered valid for training if it meets the criteria for being a valid testing instance
        and does not have any missing values.

        The method checks if:
            - The instance is valid for testing (i.e., `is_valid_testing_instance` is `True`).
            - There are no missing values (`self.has_missing` is `False`).

        Returns:
            bool: `True` if the instance is valid for training, `False` otherwise.

        """
        return self.is_valid_testing_instance and not self.has_missing
    
    @property
    def is_valid_testing_instance(self:Self)->bool:
        """
        Determines if the current instance is valid for testing purposes.

        An instance is considered valid for testing if it does not belong to the 
        categories of CADAVER or STILLBORN and if it has a non-`None` code.

        The method checks if:
            - The admission category is neither CADAVER nor STILLBORN.
            - The code attribute is not `None`.

        Returns:
            bool: `True` if the instance is valid for testing, `False` otherwise.

        """
        return self.admit_category != AdmitCategory.CADAVER and \
                self.admit_category != AdmitCategory.STILLBORN and \
                not self.code is None

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # NUMERICAL FEATURES
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    @staticmethod
    def numerical_features(admissions: list[Self],) -> pd.DataFrame:
        """
        Generates a DataFrame of numerical features for a list of `Admission` instances.

        This method extracts numerical features from each `Admission` instance and returns them as a DataFrame. 
        The numerical features include age, case mix group (CMG), case weight (or RIW), acute days, 
        and alcohol consumption days.

        The method performs the following steps:
            - Extracts the specified numerical features from each admission.
            - Constructs a DataFrame with the extracted features.
            - Ensures that the DataFrame does not contain any missing values.

        The resulting DataFrame contains columns for each of the following features:
            - `age`: The age of the patient.
            - `cmg`: The case mix group of the admission.
            - `case_weight`: The weight assigned to the case.
            - `acute_days`: The number of days classified as acute.
            - `alc_days`: The number of days related to alcohol consumption.

        Args:
            admissions (list[Self]): A list of `Admission` instances for which to generate numerical features.

        Returns:
            pd.DataFrame: A DataFrame containing the numerical features with rows corresponding to admissions and columns 
                        corresponding to the feature names (`age`, `cmg`, `case_weight`, `acute_days`, `alc_days`).

        """
        fields = ['age', 'cmg', 'case_weight', 'acute_days', 'alc_days']
        # assert all([admission.is for admission in admissions])
        vectors = []
        for admission in admissions:
            vectors.append([getattr(admission, field) for field in fields])
        matrix = np.vstack(vectors)

        df =  pd.DataFrame(matrix, columns=fields)

        # Missing from training are removed, missing from testing are fixed. 
        # Should not be na values.
        assert df.dropna().shape[0]==df.shape[0]

        return df

    @staticmethod
    def get_diagnoses_mapping():
        """
        Loads and returns the mapping of diagnoses from a JSON file specified in the configuration.

        Returns:
            dict: A dictionary mapping diagnoses codes to their descriptions.
        """
        config = configuration.get_config()
        return json.load(open(config['diagnosis_dict'], encoding='utf-8'))

    
    @staticmethod
    def get_intervention_mapping():
        """
        Loads and returns the mapping of interventions from a JSON file specified in the configuration.

        Returns:
            dict: A dictionary mapping intervention codes to their descriptions.
        """
        config = configuration.get_config()
        return json.load(open(config['intervention_dict'], encoding='utf-8'))

    @staticmethod
    def get_y(admissions: list[Self])->np.ndarray:
        """
        Generates an array of readmission indicators (target) from a list of admission objects.

        Args:
            admissions (list[Self]): A list of admission objects to process.

        Returns:
            np.ndarray: A NumPy array where each element is 1 if the admission has an unplanned readmission,
                        otherwise 0.
        """
        return np.array([1 if admission.has_readmission and \
                     admission.readmission.readmission_code!=ReadmissionCode.PLANNED_READMIT else 0 \
                     for admission in admissions])

    @property
    def has_readmission(self: Self,)->bool:
        """
        Checks whether the admission has a readmission.

        Returns:
            bool: True if the admission has a readmission, otherwise False.
        """
        return not self.readmission is None

    @property
    def length_of_stay(self: Self)->int:
        """
        Calculates the length of stay for the admission in days.

        Returns:
            int: The number of days between the admit_date and discharge_date. Returns None if
                 admit_date is not set.
        """
        los = None
        if not self.admit_date is None:
            los = (self.discharge_date - self.admit_date).days
        return los
    
    def is_valid_readmission(self, readmission: Self)->bool:
        """
        Determines if the provided readmission is valid. A readmission is considered valid if:
            - It occurs within 30 days of the original admission (self), and
            - The readmission is classified as a readmission by its readmission code.

       IMPORTANT:
        This method does not check if the readmission was planned. A readmission occurring within 30 days 
        is considered valid regardless of whether it was planned or not. However, when computing the target 
        variable using `get_y`, each admission is checked if planned or not, as our target is **unplanned**
        hospital readmission within 30 days of discharge.

        
        Args:
            readmission (Self): The readmission to be validated.

        Returns:
            bool: True if the `readmission` is valid for the admission instance (`self`), otherwise False.
        """
        return (readmission.admit_date - self.discharge_date).days<=30 and \
            ReadmissionCode.is_readmit(readmission.readmission_code)
    
    def add_readmission(self, readmission: Self):
        """
        Assigns the provided readmission as the readmission for the current admission instance (`self`).
        This method assumes that the provided readmission has been validated using `is_valid_readmission`.

        Args:
            readmission (Self): The readmission to be added.

        Raises:
            AssertionError: If the `readmission` is not valid according to `is_valid_readmission`.
        """
        assert self.is_valid_readmission(readmission)
        self.readmission = readmission



    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # GET TRAINING AND TESTING DATA
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    @staticmethod
    def get_training_testing_data(filtering=True, 
                                  combining_diagnoses=False, 
                                  combining_interventions=False) -> list[Self]:
        """
        Retrieves and prepares training and testing data for model development.

        This method performs the following steps:
        1. Loads data from a JSON file containing admissions data.
        2. Converts the JSON data into instances of the `Admission` class.
        3. Organizes the data by patient and sorts the admissions by discharge date.
        4. Identifies valid readmissions based on the readmission code and adds them to the original admission records.
        5. Splits the data into training and testing sets.
        6. Optionally filters out instances with missing values or specific admission categories.
        7. Optionally combines diagnosis and intervention information across admissions for each patient.

        Args:
            filtering (bool, optional): If True, removes instances with missing values or invalid categories from the training and testing sets. Defaults to True.
            combining_diagnoses (bool, optional): If True, combines diagnosis information across admissions for each patient. Defaults to False.
            combining_interventions (bool, optional): If True, combines intervention information across admissions for each patient. Defaults to False.

        Returns:
            tuple: A tuple containing two lists:
                - A list of `Admission` instances for the training set.
                - A list of `Admission` instances for the testing set.
        
        Raises:
            AssertionError: If the training data indices do not match expected values or the hash of the indices does not match.
        """
        rng = np.random.default_rng(seed=5348363479653547918)
        config = configuration.get_config()

        # ---------- ---------- ---------- ---------- 
        # Retriving train testing data from JSON file
        # ---------- ---------- ---------- ---------- 
        # f = open(config['train_val_json'])
        # train_val_data = json.load(f)
        train_val_data = data_manager.get_train_test_json_content()

        # ---------- ---------- ---------- ---------- 
        # Converting JSON to DataClasses
        # ---------- ---------- ---------- ---------- 
        all_admissions = []
        for ix in train_val_data:
            all_admissions.append(
                Admission.from_dict_data(admit_id=int(ix), admission=train_val_data[ix])
                )

        # ---------- ---------- ---------- ---------- 
        # Dictionary organizing data by patient
        # ---------- ---------- ---------- ---------- 
        patient2admissions = defaultdict(list)
        for admission in all_admissions:
            code = admission.code
            patient2admissions[code].append(admission)
            
        # print(set([str(admission.entry_code) for admission in all_admissions]))

        # ---------- ---------- ---------- ----------
        # Ordering patient list by discharge date (from back )
        # ---------- ---------- ---------- ----------
        for patient_code in patient2admissions:
            admissions_list = patient2admissions[patient_code]
            admissions_list = sorted(admissions_list, key=lambda admission: admission.discharge_date, reverse=False)
            assert all([admissions_list[i].discharge_date <= admissions_list[i+1].discharge_date for i in range(len(admissions_list)-1)])
            patient2admissions[patient_code] = admissions_list



        patient_count=0
        valid_readmission_count=0
        for patient_code in patient2admissions:
            patient_admissions = patient2admissions[patient_code]
            ix = 0 
            while ix < len(patient_admissions):
                readmission_code = patient_admissions[ix].readmission_code
                if ReadmissionCode.is_readmit(readmission_code):
                    # Either is not the first admission (ix>0) or 
                    # we don't have the patient previous admition (readmission close to begining of dataset) (admit-(2015-01-01))<28 days
                    # assert ix>0 or (patient_admissions[ix].admit_date - datetime.datetime.fromisoformat('2015-01-01')).days<365
                    if ix>0 and  patient_admissions[ix-1].is_valid_readmission(patient_admissions[ix]):
                        patient_admissions[ix-1].add_readmission(patient_admissions[ix])
                        valid_readmission_count+=1
                ix+=1
            patient_count+=1


        train_indexes = rng.choice(range(len(all_admissions)),size=int(0.8*len(all_admissions)), replace=False)

        # Checking that every time I am getting the same training instances ( and validation instances)
        assert all(train_indexes[:3] ==np.array([478898, 46409, 322969]))
        assert all(train_indexes[-3:] ==np.array([415014, 330673, 338415]))
        assert hash(tuple(train_indexes))==2028319680436964623

        train_indexes = set(train_indexes)

        train = [admission for ix, admission in enumerate(all_admissions) if ix in train_indexes ]
        testing = [admission for ix, admission in enumerate(all_admissions) if not ix in train_indexes ]


        # ---------- ---------- ---------- ----------
        # Filtering instances with missing values
        # ---------- ---------- ---------- ----------
        # Remove from training instances with missing values or with admit category in {CADAVER, STILLBORN}
        if filtering:
            train = list(filter(lambda admission: admission.is_valid_training_instance, train))

            # Remove from testing instances without patient code and admit category in {CADAVER, STILLBORN}
            testing = list(filter(lambda admission: admission.is_valid_testing_instance , testing))

        if combining_diagnoses or combining_interventions:
            # Grouping Admission per patient
            patient2admissions = {}
            for admission in train+testing:
                if not admission.code in patient2admissions:
                    patient2admissions[admission.code]=[]
                patient2admissions[admission.code].append(admission)

            # Sorting by date
            for code_ in patient2admissions.keys():
                patient2admissions[code_] = sorted(patient2admissions[code_], 
                                                    key=lambda admission: admission.discharge_date)

        if combining_diagnoses:
            # Combininig diagnosis
            for patient_code, patient_admissions  in patient2admissions.items():
                for ix, admission in enumerate(patient_admissions):
                    if ix>0:
                        previous_admission = patient_admissions[ix-1]
                        admission.diagnosis.prepend_diagnosis(previous_admission.diagnosis)

        if combining_interventions:
            # Combininig interventions
            for patient_code, patient_admissions  in patient2admissions.items():
                for ix, admission in enumerate(patient_admissions):
                    if ix>0:
                        previous_admission = patient_admissions[ix-1]
                        admission.intervention_code = previous_admission.intervention_code + admission.intervention_code
                        admission.px_long_text = previous_admission.px_long_text + admission.px_long_text

        return train, testing


    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    @staticmethod
    def get_train_test_matrices(params):
        """
        +-------------------------------------------------------------------------------------+
        | IMPORTANT:                                                                          |
        | For cross-validation experiments we retrieve all data and divide it into folds. For | 
        | explainability experiments we use train/test split. So, the methods to obtain train |
        | and test matrices(get_train_test_matrices) and data (get_training_testing_data) are |
        | used by all explainability methods:                                                 |
        |  - Permutation Feature Importance                                                   |
        |  - Explainable Decision Tree                                                        |
        |  - Explainable Logistic Regression                                                  |
        |  - SHAP                                                                             |
        |  - Guidelines (explainable Decision Trees)                                          |
        +-------------------------------------------------------------------------------------+

        Retrieves and prepares training and testing matrices for model development, including feature extraction,
        transformation, and sampling.

        This method performs the following steps:
        1. Retrieves training and testing data (get_training_testing_data) and handles missing values if required.
        2. Constructs training and testing feature matrices based on specified feature types (e.g., numerical, 
           categorical, diagnosis, intervention).
        3. Applies various data preprocessing techniques, including scaling, normalization, and fixing skew.
        4. Optionally combines diagnosis and intervention features based on parameters.
        5. Handles outlier removal and feature scaling.
        6. Applies over-sampling, under-sampling, or both to address class imbalance if specified.
        7. Removes constant features and performs feature selection if specified.

        Args:
            params (dict): A dictionary of parameters controlling the data preparation process. Expected keys include:
                - 'combining_diagnoses' (bool): Whether to combine diagnosis features across admissions.
                - 'combining_interventions' (bool): Whether to combine intervention features across admissions.
                - 'fix_missing_in_testing' (bool): Whether to fix missing values in the testing set based on the training set.
                - 'numerical_features' (bool): Whether to include numerical features.
                - 'categorical_features' (bool): Whether to include categorical features.
                - 'diagnosis_features' (bool): Whether to include diagnosis features.
                - 'intervention_features' (bool): Whether to include intervention features.
                - 'diagnosis_embeddings' (bool): Whether to include diagnosis embeddings.
                - 'intervention_embeddings' (bool): Whether to include intervention embeddings.
                - 'fix_skew' (bool): Whether to apply skew correction to numerical features.
                - 'normalize' (bool): Whether to normalize numerical features.
                - 'remove_outliers' (bool): Whether to remove outliers from numerical features.
                - 'min_df' (int): Minimum document frequency for diagnosis and intervention features.
                - 'use_idf' (bool): Whether to use inverse document frequency in feature extraction.
                - 'under_sample_majority_class' (bool): Whether to perform under-sampling of the majority class.
                - 'over_sample_minority_class' (bool): Whether to perform over-sampling of the minority class.
                - 'smote_and_undersampling' (bool): Whether to apply both SMOTE and under-sampling.
                - 'over_sampling_ration' (float): Ratio for over-sampling if SMOTE is used.
                - 'under_sampling_ration' (float): Ratio for under-sampling if SMOTE is used.
                - 'feature_selection' (bool): Whether to apply feature selection.
                - 'k_best_features' (int): Number of top features to select if feature selection is applied.
                - 'diag_embedding_model_name' (str): Model name for diagnosis embeddings.
                - 'interv_embedding_model_name' (str): Model name for intervention embeddings.

        Returns:
            tuple: A tuple containing:
                - X_train (sparse matrix): The feature matrix for the training set.
                - y_train (array): The target vector for the training set.
                - X_test (sparse matrix): The feature matrix for the testing set.
                - y_test (array): The target vector for the testing set.
                - columns (array): An array of feature column names corresponding to the feature matrices.

        Raises:
            AssertionError: If certain conditions regarding data sampling or feature consistency are not met.
        """
        config = configuration.get_config()

        columns = []
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # RETRIEVING TRAIN AND TEST
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        combining_diagnoses = True if 'combining_diagnoses' in params and params['combining_diagnoses'] else False 

        combining_interventions = True if 'combining_interventions' in params and params['combining_interventions'] else False 
        io.debug(f'Calling Admission.get_training_testing_date(combining_diagnoses={combining_diagnoses}, combining_interventions={combining_interventions})')
        training ,testing = Admission.get_training_testing_data(combining_diagnoses=combining_diagnoses, 
                                                                combining_interventions=combining_interventions)
        if params['fix_missing_in_testing']:
            for admission in testing:
                admission.fix_missings(training)

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # TRAINING MATRIX
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        features = []
        if params['numerical_features']:
            numerical_df = Admission.numerical_features(training,)
            columns += list(numerical_df.columns)
            if params['remove_outliers']:
                stds = np.std(numerical_df)
                mean = np.mean(numerical_df, axis=0)
                is_outlier=np.sum(numerical_df.values > (mean+4*stds).values, axis=1)>0
            
            if params['fix_skew']:
                numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
                numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
                numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

            if params['normalize']:
                scaler = StandardScaler()
                if params['remove_outliers']:
                    scaler.fit(numerical_df.values[~is_outlier,:])
                else:
                    scaler.fit(numerical_df.values)
                numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)

            features.append(sparse.csr_matrix(numerical_df.values))

        if params['categorical_features']:
            categorical_df, main_pt_services_list = Admission.categorical_features(training)
            columns += list(categorical_df.columns)
            features.append(sparse.csr_matrix(categorical_df.values))

        if params['diagnosis_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_diagnosis, diagnosis_matrix = Admission.diagnosis_codes_features(training,
                                                                                   use_idf=params['use_idf'],
                                                                                   min_df=min_df,
                                                                                  )
            features.append(diagnosis_matrix)
            columns += list(vocab_diagnosis)


        if params['intervention_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_interventions, intervention_matrix = Admission.intervention_codes_features(training,
                                                                                             min_df=min_df,
                                                                                             use_idf=params['use_idf'],
                                                                                            )
            features.append(intervention_matrix)
            columns += list(vocab_interventions)

        if 'diagnosis_embeddings' in params and params['diagnosis_embeddings']:
            io.debug(f"Loading diagnosis embeddings from model: {params['diag_embedding_model_name']}")
            # If combining diagnosis then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_diagnoses
            diagnosis_embeddings_df = Admission.diagnosis_embeddings(training,
                                                                     model_name=params['diag_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            io.debug(f"Diagnosis model loaded. Shape of diag_emb_df={diagnosis_embeddings_df.shape}")

            features.append(sparse.csr_matrix(diagnosis_embeddings_df.values))
            columns += list(diagnosis_embeddings_df.columns)

        if 'intervention_embeddings' in params and params['intervention_embeddings']:
            io.debug(f"Loading intervention embeddings from model: {params['interv_embedding_model_name']}")
            # If combining intervention then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_interventions

            intervention_embeddings_df = Admission.intervention_embeddings(training,
                                                                     model_name=params['interv_embedding_model_name'],
                                                                     use_cached=use_cached
                                                                     )
            io.debug(f"Intervention model loaded. Shape of interv_emb_df={intervention_embeddings_df.shape}")
            features.append(sparse.csr_matrix(intervention_embeddings_df.values))
            columns += list(intervention_embeddings_df.columns)

        if params['remove_outliers'] and params['numerical_features']:
            mask=~is_outlier
        else:
            mask = np.ones(shape=(len(training)))==1

        for ix, matrix in enumerate(features):
            print(f'{ix:2} matrix.shape={matrix.shape}')
        X_train = sparse.hstack([matrix[mask,:] for matrix in features])
        y_train = Admission.get_y(training)[mask]


        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # TESTING MATRIX
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        features = []
        if params['numerical_features']:
            numerical_df = Admission.numerical_features(testing,)
            
            if params['fix_skew']:
                numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
                numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
                numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

            if params['normalize']:
                numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)
            features.append(sparse.csr_matrix(numerical_df.values))

        if params['categorical_features']:
            categorical_df,_ = Admission.categorical_features(testing, main_pt_services_list=main_pt_services_list)
            features.append(sparse.csr_matrix(categorical_df.values))

        if params['diagnosis_features']:            
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_diagnosis, diagnosis_matrix = Admission.diagnosis_codes_features(testing, 
                                                                                   vocabulary=vocab_diagnosis, 
                                                                                   use_idf=params['use_idf'],
                                                                                   min_df=min_df,
                                                                                  )
            features.append(diagnosis_matrix)

        if params['intervention_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_interventions, intervention_matrix = Admission.intervention_codes_features(testing, 
                                                                                             vocabulary=vocab_interventions, 
                                                                                             use_idf=params['use_idf'],
                                                                                             min_df=min_df,
                                                                                             )
            features.append(intervention_matrix)

        if 'diagnosis_embeddings' in params and params['diagnosis_embeddings']:
            io.debug(f"Loading diagnosis embeddings from model: {params['diag_embedding_model_name']}")
            # If combining diagnosis then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_diagnoses
            diagnosis_embeddings_df = Admission.diagnosis_embeddings(testing,
                                                                     model_name=params['diag_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            io.debug(f"Model loaded, shape={diagnosis_embeddings_df.shape}")
            features.append(sparse.csr_matrix(diagnosis_embeddings_df.values))
        

        if 'intervention_embeddings' in params and params['intervention_embeddings']:
            # If combining intervention then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_interventions
            intervention_embeddings_df = Admission.intervention_embeddings(testing,
                                                                     model_name=params['interv_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            io.debug(f"Model loaded, shape={intervention_embeddings_df.shape}")
            features.append(sparse.csr_matrix(intervention_embeddings_df.values))

        X_test = sparse.hstack(features)
        y_test = Admission.get_y(testing)

        columns = np.array(columns)

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # OVER or UNDER SAMPLING (CHANGING NUMBER OF INSTANCES):
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

        SAMPLING_SEED = 1270833263
        sampling_random_state = np.random.RandomState(SAMPLING_SEED)

        if params['under_sample_majority_class']:
            assert not params['over_sample_minority_class']
            assert not params['smote_and_undersampling']

            under_sampler = RandomUnderSampler(sampling_strategy=1, random_state=sampling_random_state)
            io.debug('Under Sampling training set before calling fit ....')

            # Under sampling:
            X_train, y_train = under_sampler.fit_resample(X_train, y_train)
            io.debug(f'resampled(X_train).shape = {X_train.shape}')
            io.debug(f'resampled(y_train).shape = {y_train.shape}')

        elif params['over_sample_minority_class']:
            assert not params['under_sample_majority_class']
            assert not params['smote_and_undersampling']

            over_sample = SMOTE(sampling_strategy=1, random_state=sampling_random_state)
            io.debug('Over Sampling training set before calling fit ....')

            X_train, y_train = over_sample.fit_resample(X_train, y_train)
            io.debug(f'resampled(X_train).shape = {X_train.shape}')
            io.debug(f'resampled(y_train).shape = {y_train.shape}')

        elif params['smote_and_undersampling']:
            assert not params['under_sample_majority_class']
            assert not params['over_sample_minority_class']

            over = SMOTE(sampling_strategy=params['over_sampling_ration'], 
                         random_state=sampling_random_state
                         )
            under = RandomUnderSampler(sampling_strategy=params['under_sampling_ration'], 
                                       random_state=sampling_random_state
                                       )
            
            steps = [('o', over), 
                     ('u', under)]
            
            pipeline = Pipeline(steps=steps)
            io.debug('Applying both under and over sampling ....')

            X_train, y_train = pipeline.fit_resample(X_train, y_train)
            io.debug(f'resampled(X_train).shape = {X_train.shape}')
            io.debug(f'resampled(y_train).shape = {y_train.shape}')

        else:
            io.debug('Using X_train, y_train, no samplig strategy ...')



        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # REMOVING CONSTANT VARIABLES (CHANGING NUMBER OF COLUMNS, need to update all matrices)
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        io.debug('Looking for constant variables ...')
        columns = np.array(columns) 

        io.debug('Using memory efficient solution')
        constant_variables = np.array(list(
            map(lambda ix: True if np.var(X_train[:,ix].toarray())==0 else False, range(X_train.shape[1]))
        ))


        if np.sum(constant_variables)>0:
            # X = X[:,~constant_variables]
            X_train = X_train[:,~constant_variables]
            X_test = X_test[:,~constant_variables]
            columns = columns[~constant_variables]
            io.debug(f'Removed {np.sum(constant_variables)} columns')
        else:
            io.debug('Not constant variables found ...')

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # FEATURE SELECTION (CHANING COLUMNS, NEED TO UDPATE ALL MATRICES)
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        io.debug('Shapes of matrices before FS...')
        io.debug(f'X_train: {X_train.shape}')
        io.debug(f'y_train: {y_train.shape}')
        io.debug(f'X_test:  {X_test.shape}')
        io.debug(f'y_test:  {y_test.shape}')

        if 'feature_selection' in params and params['feature_selection']:
            io.debug('Applying feature selection')
            clf = SelectKBest(f_classif, k=params['k_best_features'], ).fit(X_train, y_train)
            X_train = clf.transform(X_train)
            X_test = clf.transform(X_test)
            columns = clf.transform(columns.reshape(1,-1))[0,:]

        return X_train, y_train, X_test, y_test, columns

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # FIX MISSINGS
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    def fix_missings(self: Self, training: list[Self]):
        """
        Imputes missing values for an admission entry based on the provided training data.

        This method addresses various missing attributes of the admission entry using statistical
        imputation techniques and random sampling from the training data. It modifies the instance
        attributes in place.

        The following missing values are handled:
        - **admit_date**: Estimated using the average and standard deviation of length of stay from the training data.
        - **case_weight**: Imputed using the average and standard deviation of case weights from the training data.
        - **gender**: Assigned randomly from the training data if missing.
        - **admit_category**: Assigned randomly from the training data if missing.
        - **readmission_code**: Assigned randomly from the training data if missing.
        - **transfusion_given**: Assigned randomly from the training data if missing.
        - **cmg**: Imputed using a random value (uniform) within the range of existing CMG values from the training data.
        - **main_pt_service**: Set to '<NONE>' if missing.
        - **mrdx**: Set to '<NONE>' if missing.
        - **entry_code**: Assigned randomly from the training data if missing.

        Args:
            training (list[Self]): A list of training admission entries used to impute missing values.

        Raises:
            AssertionError: If the `code` attribute of the instance is `None`, indicating that the target variable cannot be recovered.
        """
        rng = np.random.default_rng(seed=5348363479653547918)
        assert not self.code is None, 'Cannot fix an entry without code (cannot recover target variable without it).'

        if self.admit_date is None:
            avg_los = np.average([admission.length_of_stay for admission in training if not admission.length_of_stay is None])
            std_los = np.std([admission.length_of_stay for admission in training if not admission.length_of_stay is None])
            los = int(rng.normal(loc=avg_los, scale=std_los, size=1)[0])

            self.admit_date = self.discharge_date - datetime.timedelta(days=los)

        if self.case_weight is None or np.isnan(self.case_weight):
            avg_case_weight =  np.average([admission.case_weight for admission in training if not admission.case_weight is None and not np.isnan(admission.case_weight)])
            std_case_weight =  np.std([admission.case_weight for admission in training if not admission.case_weight is None and not np.isnan(admission.case_weight)])

            self.case_weight = rng.normal(loc=avg_case_weight, scale=std_case_weight, size=1)[0]

        if self.gender  == Gender.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.gender = training[ix].gender

        if self.admit_category == AdmitCategory.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.admit_category = training[ix].admit_category
            
        if self.readmission_code == ReadmissionCode.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.readmission_code = training[ix].readmission_code

        if self.transfusion_given == TransfusionGiven.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.transfusion_given = training[ix].transfusion_given

        if self.cmg is None or np.isnan(self.cmg):
            new_cmb = rng.uniform(low=min([admission.cmg for admission in training if not admission.cmg is None]), 
                                high=max([admission.cmg for admission in training if not admission.cmg is None]), 
                                size=1)[0]
            self.cmg = new_cmb

        if self.main_pt_service is None:
            self.main_pt_service = '<NONE>'

        if self.mrdx is None:
            self.mrdx = '<NONE>'

        if self.entry_code  == EntryCode.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.entry_code = training[ix].entry_code

    # AFTER CV CORRECTIONS
    @staticmethod
    def get_train_test_data(filtering=True,
                                  combining_diagnoses=False, 
                                  combining_interventions=False) -> list[Self]:
        """
        +------------------------------------------------------------+
        | WHERE IS USED                                              |
        |                                                            |
        | This method retrieves and processes both the training and  |
        | testing data (collectively referred to as the development  |
        | set) and returns them as a single collection. It is        |
        | primarily used to obtain the development set for           |
        | cross-validation purposes. Specifically, this method is    |
        | crucial for:                                               |
        |                                                            |
        | - get_both_train_test_matrices, which is further utilized  |
        |   in:                                                      |
        |     * build_dataset_statistics_table                       |
        |     * running_all_experiments_cv                           |
        |                                                            |
        | By combining the training and testing data into a unified  |
        | development set, this approach facilitates their           |
        | subsequent separation into folds for cross-validation.     |
        |                                                            |
        | Because it uses train and test together, it filters all    |
        | data as if they were testing instances                     |
        | (`is_valid_testing_instance`) and not training instances   |
        | (`is_valid_training_instance`), as the latter is more      |
        | restrictive and filters out more instances.                |
        +------------------------------------------------------------+

        This method loads admission train and test data from a JSON file, processes it into a
        structured format, and optionally applies various filtering and combination
        techniques to prepare the data for machine learning tasks.

        Args:
            filtering (bool, optional): If True, filters out invalid testing instances
                (e.g., admissions with null patient codes or inadmissible categories (STILLBORN, CADAVER)).
                Defaults to True.
            combining_diagnoses (bool, optional): If True, combines diagnoses from 
                previous admissions within the same patient. Defaults to False.
            combining_interventions (bool, optional): If True, combines interventions 
                from previous admissions within the same patient. Defaults to False.

        Returns:
            list[Self]: A list of processed `Admission` instances (from the development set)
                        ready for posterior computations such as cross-validation or building 
                        dataset statistics. 
        """
        rng = np.random.default_rng(seed=5348363479653547918)
        config = configuration.get_config()

        # ---------- ---------- ---------- ---------- 
        # Retriving train testing data from JSON file
        # ---------- ---------- ---------- ---------- 
        train_val_data = data_manager.get_train_test_json_content()

        # ---------- ---------- ---------- ---------- 
        # Converting JSON to DataClasses
        # ---------- ---------- ---------- ---------- 
        all_admissions = []
        for ix in train_val_data:
            all_admissions.append(
                Admission.from_dict_data(admit_id=int(ix), admission=train_val_data[ix])
                )

        # ---------- ---------- ---------- ---------- 
        # Dictionary organizing data by patient
        # ---------- ---------- ---------- ---------- 
        patient2admissions = defaultdict(list)
        for admission in all_admissions:
            code = admission.code
            patient2admissions[code].append(admission)
            

        # ---------- ---------- ---------- ----------
        # Ordering patient list by discharge date (from back )
        # ---------- ---------- ---------- ----------
        for patient_code in patient2admissions:
            admissions_list = patient2admissions[patient_code]
            admissions_list = sorted(admissions_list, 
                                     key=lambda admission: admission.discharge_date, 
                                     reverse=False)
            assert all([admissions_list[i].discharge_date <= admissions_list[i+1].discharge_date for i in range(len(admissions_list)-1)])
            patient2admissions[patient_code] = admissions_list

        # print(set([str(admission.entry_code) for admission in all_admissions]))

        patient_count=0
        valid_readmission_count=0
        for patient_code in patient2admissions:
            patient_admissions = patient2admissions[patient_code]
            ix = 0 
            while ix < len(patient_admissions):
                readmission_code = patient_admissions[ix].readmission_code
                if ReadmissionCode.is_readmit(readmission_code):
                    # Either is not the first admission (ix>0) or 
                    # we don't have the patient previous admition (readmission close to begining of dataset) (admit-(2015-01-01))<28 days
                    # assert ix>0 or (patient_admissions[ix].admit_date - datetime.datetime.fromisoformat('2015-01-01')).days<365
                    if ix>0 and  patient_admissions[ix-1].is_valid_readmission(patient_admissions[ix]):
                        patient_admissions[ix-1].add_readmission(patient_admissions[ix])
                        valid_readmission_count+=1
                ix+=1
            patient_count+=1

        # ---------- ---------- ---------- ----------
        # Filtering instances with missing values
        # ---------- ---------- ---------- ----------
        # Remove from training instances with null patient code or with admit category in {CADAVER, STILLBORN}
        if filtering:
            all_admissions = list(filter(lambda admission: admission.is_valid_testing_instance, all_admissions))

        if combining_diagnoses or combining_interventions:
            # Grouping Admission per patient
            patient2admissions = {}
            for admission in all_admissions:
                if not admission.code in patient2admissions:
                    patient2admissions[admission.code]=[]
                patient2admissions[admission.code].append(admission)

            # Sorting by date
            for code_ in patient2admissions.keys():
                patient2admissions[code_] = sorted(patient2admissions[code_], 
                                                    key=lambda admission: admission.discharge_date)

        if combining_diagnoses:
            # Combininig diagnosis
            for patient_code, patient_admissions  in patient2admissions.items():
                for ix, admission in enumerate(patient_admissions):
                    if ix>0:
                        previous_admission = patient_admissions[ix-1]
                        admission.diagnosis.prepend_diagnosis(previous_admission.diagnosis)

        if combining_interventions:
            # Combininig interventions
            for patient_code, patient_admissions  in patient2admissions.items():
                for ix, admission in enumerate(patient_admissions):
                    if ix>0:
                        previous_admission = patient_admissions[ix-1]
                        admission.intervention_code = previous_admission.intervention_code + admission.intervention_code
                        admission.px_long_text = previous_admission.px_long_text + admission.px_long_text

        return all_admissions
    
    @staticmethod
    def get_heldout_data(filtering=True,
                         combining_diagnoses=False, 
                         combining_interventions=False) -> list[Self]:
        """
        Retrieves and processes the held-out data, converting it from JSON format 
        into a list of Admission instances. The method allows for optional filtering 
        and combination of diagnoses and interventions across patient admissions.

        Args:
            filtering (bool, optional): If True, filters out invalid instances
                (is_valid_testing_instance), such as those with missing values 
                or improper admit categories (STILLBORN, CADAVER). 
                Defaults to True.
            combining_diagnoses (bool, optional): If True, combines diagnoses 
                from previous admissions for each patient. Defaults to False.
            combining_interventions (bool, optional): If True, combines interventions 
                from previous admissions for each patient. Defaults to False.

        Returns:
            list[Self]: A list of Admission instances representing the held-out data 
            after processing, filtering, and optional combination of diagnoses 
            and interventions.
        """

        rng = np.random.default_rng(seed=5348363479653547918)
        config = configuration.get_config()

        # ---------- ---------- ---------- ---------- 
        # Retriving train testing data from JSON file
        # ---------- ---------- ---------- ---------- 

        heldout_data = data_manager.get_heldout_json_content()

        # ---------- ---------- ---------- ---------- 
        # Converting JSON to DataClasses
        # ---------- ---------- ---------- ---------- 
        all_admissions = []
        for ix in heldout_data:
            all_admissions.append(
                Admission.from_dict_data(admit_id=int(ix), admission=heldout_data[ix])
                )

        # ---------- ---------- ---------- ---------- 
        # Dictionary organizing data by patient
        # ---------- ---------- ---------- ---------- 
        patient2admissions = defaultdict(list)
        for admission in all_admissions:
            code = admission.code
            patient2admissions[code].append(admission)
            
        # print(set([str(admission.entry_code) for admission in all_admissions]))

        # ---------- ---------- ---------- ----------
        # Ordering patient list by discharge date (from back )
        # ---------- ---------- ---------- ----------
        for patient_code in patient2admissions:
            admissions_list = patient2admissions[patient_code]
            admissions_list = sorted(admissions_list, 
                                     key=lambda admission: admission.discharge_date, 
                                     reverse=False)
            assert all([admissions_list[i].discharge_date <= admissions_list[i+1].discharge_date for i in range(len(admissions_list)-1)])
            patient2admissions[patient_code] = admissions_list

        # print(set([str(admission.entry_code) for admission in all_admissions]))

        patient_count=0
        valid_readmission_count=0
        for patient_code in patient2admissions:
            patient_admissions = patient2admissions[patient_code]
            ix = 0 
            while ix < len(patient_admissions):
                readmission_code = patient_admissions[ix].readmission_code
                if ReadmissionCode.is_readmit(readmission_code):
                    # Either is not the first admission (ix>0) or 
                    # we don't have the patient previous admition (readmission close to begining of dataset) (admit-(2015-01-01))<28 days
                    # assert ix>0 or (patient_admissions[ix].admit_date - datetime.datetime.fromisoformat('2015-01-01')).days<365
                    if ix>0 and  patient_admissions[ix-1].is_valid_readmission(patient_admissions[ix]):
                        patient_admissions[ix-1].add_readmission(patient_admissions[ix])
                        valid_readmission_count+=1
                ix+=1
            patient_count+=1

        # ---------- ---------- ---------- ----------
        # Filtering instances with missing values
        # ---------- ---------- ---------- ----------
        # Remove from training instances with null patient code or with admit category in {CADAVER, STILLBORN}
        if filtering:
            all_admissions = list(filter(lambda admission: admission.is_valid_testing_instance, all_admissions))

        if combining_diagnoses or combining_interventions:
            # Grouping Admission per patient
            patient2admissions = {}
            for admission in all_admissions:
                if not admission.code in patient2admissions:
                    patient2admissions[admission.code]=[]
                patient2admissions[admission.code].append(admission)

            # Sorting by date
            for code_ in patient2admissions.keys():
                patient2admissions[code_] = sorted(patient2admissions[code_], 
                                                    key=lambda admission: admission.discharge_date)

        if combining_diagnoses:
            # Combininig diagnosis
            for patient_code, patient_admissions  in patient2admissions.items():
                for ix, admission in enumerate(patient_admissions):
                    if ix>0:
                        previous_admission = patient_admissions[ix-1]
                        admission.diagnosis.prepend_diagnosis(previous_admission.diagnosis)

        if combining_interventions:
            # Combininig interventions
            for patient_code, patient_admissions  in patient2admissions.items():
                for ix, admission in enumerate(patient_admissions):
                    if ix>0:
                        previous_admission = patient_admissions[ix-1]
                        admission.intervention_code = previous_admission.intervention_code + admission.intervention_code
                        admission.px_long_text = previous_admission.px_long_text + admission.px_long_text

        return all_admissions

    ## BOTH MATRICES AFTER CV
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    @staticmethod
    def get_both_train_test_matrices(params):
        """
        Generates the training and testing matrices (X, y) along with feature columns based on the specified parameters.

        This method obtains both train and test splits (collectively referred to as the development set), and creates
        a input matrix (X) a target (y), and also returns the columns of the input matriz (columns). Which input features
        are used depends on the configuration of the experiments `params` (it could include some subset of features,
        normalization, feature selection, etc.).

        +--------------------------------------------------------------------------+
        |                                                                          |
        |    WHERE IS USED:                                                        |
        |    This method retrieves the entire development set (train and test) to  |
        |    facilitate their subsequent separation into folds                     |
        |    for cross-validation. It is used for:                                 |
        |        * build_dataset_statistics_table                                  |
        |        * running_all_experiments_cv                                      |
        |                                                                          |
        +--------------------------------------------------------------------------+

        +------------------------------------------------------------+
        | **IMPORTANT:**                                             |
        |                                                            |
        | Since this method is used for cross-validation,            |
        | transformations that should only be applied to             |
        | training matrices cannot be computed here. This is         |
        | because the development set will later be split into       |
        | multiple folds, each with its own train-test split.        |
        | As a result, the responsibility for performing             |
        | feature selection, oversampling, and undersampling         |
        | is delegated to the method that calls this one. In the     |
        | `running_all_experiments_cv.py` method, we generate the    |
        | development split, divide it into folds, and then, for     |
        | each fold's training set, we (optionally) perform          |
        | sampling, oversampling, and feature selection.             |
        |                                                            |
        | This is an important distinction, as methods like          |
        | `get_train_test_matrices`, which are very similar to       |
        | this one, can apply oversampling/undersampling and         |
        | feature selection directly. This is possible because       |
        | the specific parts of the data that will serve as          |
        | training and test sets are already known, eliminating      |
        | the need to defer these operations.                        |
        +------------------------------------------------------------+

        
        This method prepares feature matrices by combining various types of features such as numerical, categorical,
        diagnosis codes, intervention codes, and/or their respective embeddings. It also handles missing data, removes
        outliers, normalizes data, and applies feature selection to remove constant variables.

        Args:
            params (dict): A dictionary containing parameters for the feature extraction and processing. The keys in
                the dictionary control the following behavior:
                - 'combining_diagnoses' (bool): Whether to combine diagnosis codes.
                - 'combining_interventions' (bool): Whether to combine intervention codes.
                - 'fix_missing_in_testing' (bool): Whether to impute missing values (it fixes missing for the entire dataset)
                - 'numerical_features' (bool): Whether to include numerical features.
                - 'categorical_features' (bool): Whether to include categorical features.
                - 'diagnosis_features' (bool): Whether to include diagnosis code features.
                - 'intervention_features' (bool): Whether to include intervention code features.
                - 'remove_outliers' (bool): Whether to remove outliers based on standard deviation.
                - 'fix_skew' (bool): Whether to apply log transformation to skewed numerical features.
                - 'normalize' (bool): Whether to normalize numerical features.
                - 'use_idf' (bool): Whether to use inverse document frequency (IDF) weighting for diagnosis and intervention features.
                - 'min_df' (int): Minimum document frequency for diagnosis and intervention features (optional).
                - 'diagnosis_embeddings' (bool): Whether to include diagnosis embeddings as features.
                - 'diag_embedding_model_name' (str): Name of the model to use for generating diagnosis embeddings (required if 'diagnosis_embeddings' is True).
                - 'intervention_embeddings' (bool): Whether to include intervention embeddings as features.
                - 'interv_embedding_model_name' (str): Name of the model to use for generating intervention embeddings (required if 'intervention_embeddings' is True).

        Returns:
            tuple:
                - X (scipy.sparse.csr_matrix): The feature matrix after combining all selected features.
                - y (np.ndarray): The target variable vector.
                - columns (np.ndarray): An array of feature names corresponding to the columns in the feature matrix.

        Raises:
            KeyError: If required parameters for embeddings are not provided when embeddings are requested.
            ValueError: If any unexpected or incompatible parameters are encountered.
        """
        config = configuration.get_config()

        columns = []


        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # RETRIEVING TRAIN AND TEST
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        combining_diagnoses = True if 'combining_diagnoses' in params and params['combining_diagnoses'] else False 
        combining_interventions = True if 'combining_interventions' in params and params['combining_interventions'] else False 
        
        io.debug(f'Calling Admission.get_training_testing_date(combining_diagnoses={combining_diagnoses}, combining_interventions={combining_interventions})')
        all_admissions = Admission.get_train_test_data(combining_diagnoses=combining_diagnoses,
                                                combining_interventions=combining_interventions)
        

        if params['fix_missing_in_testing']:
            for admission in all_admissions:
                admission.fix_missings(all_admissions)

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # TRAINING MATRIX
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        features = []
        if params['numerical_features']:
            numerical_df = Admission.numerical_features(all_admissions,)
            columns += list(numerical_df.columns)
            if params['remove_outliers']:
                stds = np.std(numerical_df)
                mean = np.mean(numerical_df, axis=0)
                is_outlier=np.sum(numerical_df.values > (mean+4*stds).values, axis=1)>0
            
            if params['fix_skew']:
                numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
                numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
                numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

            if params['normalize']:
                scaler = StandardScaler()
                if params['remove_outliers']:
                    scaler.fit(numerical_df.values[~is_outlier,:])
                else:
                    scaler.fit(numerical_df.values)
                numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)

            features.append(sparse.csr_matrix(numerical_df.values))

        if params['categorical_features']:
            categorical_df, main_pt_services_list = Admission.categorical_features(all_admissions)
            columns += list(categorical_df.columns)
            features.append(sparse.csr_matrix(categorical_df.values))

        if params['diagnosis_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_diagnosis, diagnosis_matrix = Admission.diagnosis_codes_features(all_admissions,
                                                                                   use_idf=params['use_idf'],
                                                                                   min_df=min_df,
                                                                                  )
            features.append(diagnosis_matrix)
            columns += list(vocab_diagnosis)


        if params['intervention_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_interventions, intervention_matrix = Admission.intervention_codes_features(all_admissions,
                                                                                             min_df=min_df,
                                                                                             use_idf=params['use_idf'],
                                                                                            )
            features.append(intervention_matrix)
            columns += list(vocab_interventions)

        if 'diagnosis_embeddings' in params and params['diagnosis_embeddings']:
            io.debug(f"Loading diagnosis embeddings from model: {params['diag_embedding_model_name']}")
            # If combining diagnosis then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_diagnoses
            diagnosis_embeddings_df = Admission.diagnosis_embeddings(all_admissions,
                                                                     model_name=params['diag_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            io.debug(f"Diagnosis model loaded. Shape of diag_emb_df={diagnosis_embeddings_df.shape}")

            features.append(sparse.csr_matrix(diagnosis_embeddings_df.values))
            columns += list(diagnosis_embeddings_df.columns)

        if 'intervention_embeddings' in params and params['intervention_embeddings']:
            io.debug(f"Loading intervention embeddings from model: {params['interv_embedding_model_name']}")
            # If combining intervention then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_interventions

            intervention_embeddings_df = Admission.intervention_embeddings(all_admissions,
                                                                     model_name=params['interv_embedding_model_name'],
                                                                     use_cached=use_cached
                                                                     )
            io.debug(f"Intervention model loaded. Shape of interv_emb_df={intervention_embeddings_df.shape}")
            features.append(sparse.csr_matrix(intervention_embeddings_df.values))
            columns += list(intervention_embeddings_df.columns)

        if params['remove_outliers'] and params['numerical_features']:
            mask=~is_outlier
        else:
            mask = np.ones(shape=(len(all_admissions)))==1

        for ix, matrix in enumerate(features):
            print(f'{ix:2} matrix.shape={matrix.shape}')
        X = sparse.hstack([matrix[mask,:] for matrix in features])
        y = Admission.get_y(all_admissions)[mask]


        columns = np.array(columns)




        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # REMOVING CONSTANT VARIABLES (CHANGING NUMBER OF COLUMNS, need to update all matrices)
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        io.debug('Looking for constant variables ...')
        columns = np.array(columns) 


        io.debug('Using memory efficient solution')
        constant_variables = np.array(list(
            map(lambda ix: True if np.var(X[:,ix].toarray())==0 else False, range(X.shape[1]))
        ))


        if np.sum(constant_variables)>0:
            # X = X[:,~constant_variables]
            X = X[:,~constant_variables]
            columns = columns[~constant_variables]
            io.debug(f'Removed {np.sum(constant_variables)} columns')
        else:
            io.debug('Not constant variables found ...')



        return X, y, columns
    




    @staticmethod
    def get_development_and_held_out_matrices(params):
        """
        This method generates feature matrices and labels for both development and held-out datasets, with various 
        preprocessing steps such as missing value imputation, outlier removal, normalization, embedding generation, 
        and **feature selection**. It also supports **different sampling strategies** for handling class imbalance.

        Parameters:
        -----------
        params : dict
            A dictionary containing the following keys:
            
            - combining_diagnoses (bool): If True, combines diagnosis data across admissions.
            - combining_interventions (bool): If True, combines intervention data across admissions.
            - fix_missing_in_testing (bool): If True, fills in missing values in the held-out set.
            - numerical_features (bool): If True, includes numerical features.
            - categorical_features (bool): If True, includes categorical features.
            - diagnosis_features (bool): If True, includes diagnosis code features.
            - intervention_features (bool): If True, includes intervention code features.
            - diagnosis_embeddings (bool): If True, includes diagnosis embeddings.
            - intervention_embeddings (bool): If True, includes intervention embeddings.
            - remove_outliers (bool): If True, removes outliers from numerical features.
            - fix_skew (bool): If True, applies log transformation to correct skewness in numerical features.
            - normalize (bool): If True, normalizes numerical features.
            - under_sample_majority_class (bool): If True, applies undersampling to the majority class in the development set.
            - over_sample_minority_class (bool): If True, applies oversampling to the minority class in the development set.
            - smote_and_undersampling (bool): If True, applies SMOTE and undersampling to balance classes.
            - feature_selection (bool): If True, applies feature selection to reduce the number of features.
            - k_best_features (int): The number of top features to select if feature selection is enabled.
            - diag_embedding_model_name (str): Name of the model used for generating diagnosis embeddings.
            - interv_embedding_model_name (str): Name of the model used for generating intervention embeddings.
            - use_idf (bool): If True, uses TF-IDF weighting for diagnosis and intervention code features.
            - min_df (int): Minimum document frequency for including a term in the feature set.
            - over_sampling_ration (float): Sampling ratio for oversampling the minority class.
            - under_sampling_ration (float): Sampling ratio for undersampling the majority class.

        Returns:
        --------
        X_development : sparse matrix
            The feature matrix for the development set.
        
        y_development : numpy array
            The labels for the development set.
        
        X_heldout : sparse matrix
            The feature matrix for the held-out set.
        
        y_heldout : numpy array
            The labels for the held-out set.
        
        columns : numpy array
            The names of the features included in the matrices.

        WHERE IS USED:
        --------------
        It is used to test the performance of the models and explainability tools in the held-out:
         - ML_experiments_on_heldout.py and
         - test_guidelines.py
        
        """
        config = configuration.get_config()
        columns = []
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # RETRIEVING TRAIN AND TEST
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        combining_diagnoses = True if 'combining_diagnoses' in params and params['combining_diagnoses'] else False 

        combining_interventions = True if 'combining_interventions' in params and params['combining_interventions'] else False 
        print(f'Calling Admission.get_development_and_held_out_matrices(combining_diagnoses={combining_diagnoses}, combining_interventions={combining_interventions})')
        
        
        training, testing = Admission.get_training_testing_data(combining_diagnoses=combining_diagnoses, 
                                                                combining_interventions=combining_interventions)
        



        development = training+testing

        development = list(filter(lambda admission: admission.is_valid_training_instance, development))



        heldout = Admission.get_heldout_data(combining_diagnoses=combining_diagnoses, 
                                             combining_interventions=combining_interventions)
        
        
        if params['fix_missing_in_testing']:
            for admission in heldout:
                admission.fix_missings(development)

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # TRAINING MATRIX
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        features = []
        if params['numerical_features']:
            numerical_df = Admission.numerical_features(development,)
            columns += list(numerical_df.columns)
            if params['remove_outliers']:
                stds = np.std(numerical_df)
                mean = np.mean(numerical_df, axis=0)
                is_outlier=np.sum(numerical_df.values > (mean+4*stds).values, axis=1)>0
            
            if params['fix_skew']:
                numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
                numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
                numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

            if params['normalize']:
                scaler = StandardScaler()
                if params['remove_outliers']:
                    scaler.fit(numerical_df.values[~is_outlier,:])
                else:
                    scaler.fit(numerical_df.values)
                numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)

            features.append(sparse.csr_matrix(numerical_df.values))

        if params['categorical_features']:
            categorical_df, main_pt_services_list = Admission.categorical_features(development)
            columns += list(categorical_df.columns)
            features.append(sparse.csr_matrix(categorical_df.values))

        if params['diagnosis_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_diagnosis, diagnosis_matrix = Admission.diagnosis_codes_features(development,
                                                                                   use_idf=params['use_idf'],
                                                                                   min_df=min_df,
                                                                                  )
            features.append(diagnosis_matrix)
            columns += list(vocab_diagnosis)


        if params['intervention_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_interventions, intervention_matrix = Admission.intervention_codes_features(development,
                                                                                             min_df=min_df,
                                                                                             use_idf=params['use_idf'],
                                                                                            )
            features.append(intervention_matrix)
            columns += list(vocab_interventions)

        if 'diagnosis_embeddings' in params and params['diagnosis_embeddings']:
            print(f"Loading diagnosis embeddings from model: {params['diag_embedding_model_name']}")
            # If combining diagnosis then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_diagnoses
            diagnosis_embeddings_df = Admission.diagnosis_embeddings(development,
                                                                     model_name=params['diag_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            print(f"Diagnosis model loaded. Shape of diag_emb_df={diagnosis_embeddings_df.shape}")

            features.append(sparse.csr_matrix(diagnosis_embeddings_df.values))
            columns += list(diagnosis_embeddings_df.columns)

        if 'intervention_embeddings' in params and params['intervention_embeddings']:
            print(f"Loading intervention embeddings from model: {params['interv_embedding_model_name']}")
            # If combining intervention then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_interventions

            intervention_embeddings_df = Admission.intervention_embeddings(development,
                                                                     model_name=params['interv_embedding_model_name'],
                                                                     use_cached=use_cached
                                                                     )
            print(f"Intervention model loaded. Shape of interv_emb_df={intervention_embeddings_df.shape}")
            features.append(sparse.csr_matrix(intervention_embeddings_df.values))
            columns += list(intervention_embeddings_df.columns)

        if params['remove_outliers'] and params['numerical_features']:
            mask=~is_outlier
        else:
            mask = np.ones(shape=(len(development)))==1

        for ix, matrix in enumerate(features):
            print(f'{ix:2} matrix.shape={matrix.shape}')
        X_development = sparse.hstack([matrix[mask,:] for matrix in features])
        y_development = Admission.get_y(development)[mask]


        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # TESTING MATRIX
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        features = []
        if params['numerical_features']:
            numerical_df = Admission.numerical_features(heldout,)
            
            if params['fix_skew']:
                numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
                numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
                numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

            if params['normalize']:
                numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)
            features.append(sparse.csr_matrix(numerical_df.values))

        if params['categorical_features']:
            categorical_df,_ = Admission.categorical_features(heldout, 
                                                              main_pt_services_list=main_pt_services_list)
            features.append(sparse.csr_matrix(categorical_df.values))

        if params['diagnosis_features']:            
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_diagnosis, diagnosis_matrix = Admission.diagnosis_codes_features(heldout, 
                                                                                   vocabulary=vocab_diagnosis, 
                                                                                   use_idf=params['use_idf'],
                                                                                   min_df=min_df,
                                                                                  )
            features.append(diagnosis_matrix)

        if params['intervention_features']:
            min_df = params['min_df'] if 'min_df' in params else 1
            vocab_interventions, intervention_matrix = Admission.intervention_codes_features(heldout, 
                                                                                             vocabulary=vocab_interventions, 
                                                                                             use_idf=params['use_idf'],
                                                                                             min_df=min_df,
                                                                                             )
            features.append(intervention_matrix)

        if 'diagnosis_embeddings' in params and params['diagnosis_embeddings']:
            print(f"Loading diagnosis embeddings from model: {params['diag_embedding_model_name']}")
            # If combining diagnosis then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_diagnoses
            diagnosis_embeddings_df = Admission.diagnosis_embeddings(heldout,
                                                                     model_name=params['diag_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            print(f"Model loaded, shape={diagnosis_embeddings_df.shape}")
            features.append(sparse.csr_matrix(diagnosis_embeddings_df.values))
        

        if 'intervention_embeddings' in params and params['intervention_embeddings']:
            # If combining intervention then cannot use cached (cached embeddings are not combined)
            use_cached = not combining_interventions
            intervention_embeddings_df = Admission.intervention_embeddings(heldout,
                                                                     model_name=params['interv_embedding_model_name'],
                                                                     use_cached=use_cached,
                                                                     )
            print(f"Model loaded, shape={intervention_embeddings_df.shape}")
            features.append(sparse.csr_matrix(intervention_embeddings_df.values))

        X_heldout = sparse.hstack(features)
        y_heldout = Admission.get_y(heldout)

        columns = np.array(columns)


        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
        # OVER or UNDER SAMPLING (CHANGING NUMBER OF INSTANCES):
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

        SAMPLING_SEED = 1270833263
        sampling_random_state = np.random.RandomState(SAMPLING_SEED)

        if params['under_sample_majority_class']:
            assert not params['over_sample_minority_class']
            assert not params['smote_and_undersampling']

            under_sampler = RandomUnderSampler(sampling_strategy=1, random_state=sampling_random_state)
            print('Under Sampling development set before calling fit ....')

            # Under sampling:
            X_development, y_development = under_sampler.fit_resample(X_development, y_development)
            print(f'resampled(X_development).shape = {X_development.shape}')
            print(f'resampled(y_development).shape = {y_development.shape}')

        elif params['over_sample_minority_class']:
            assert not params['under_sample_majority_class']
            assert not params['smote_and_undersampling']

            over_sample = SMOTE(sampling_strategy=1, random_state=sampling_random_state)
            print('Over Sampling development set before calling fit ....')

            X_development, y_development = over_sample.fit_resample(X_development, y_development)
            print(f'resampled(X_development).shape = {X_development.shape}')
            print(f'resampled(y_development).shape = {y_development.shape}')

        elif params['smote_and_undersampling']:
            assert not params['under_sample_majority_class']
            assert not params['over_sample_minority_class']

            over = SMOTE(sampling_strategy=params['over_sampling_ration'], 
                         random_state=sampling_random_state
                         )
            under = RandomUnderSampler(sampling_strategy=params['under_sampling_ration'], 
                                       random_state=sampling_random_state
                                       )
            
            steps = [('o', over), 
                     ('u', under)]
            
            pipeline = Pipeline(steps=steps)
            print('Applying both under and over sampling ....')

            X_development, y_development = pipeline.fit_resample(X_development, y_development)
            print(f'resampled(X_development).shape = {X_development.shape}')
            print(f'resampled(y_development).shape = {y_development.shape}')

        else:
            print('Using X_development, y_development, no samplig strategy ...')



        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # REMOVING CONSTANT VARIABLES (CHANGING NUMBER OF COLUMNS, need to update all matrices)
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        print('Looking for constant variables ...')
        columns = np.array(columns) 

        print('Using memory efficient solution')
        constant_variables = np.array(list(
            map(lambda ix: True if np.var(X_development[:,ix].toarray())==0 else False, range(X_development.shape[1]))
        ))


        if np.sum(constant_variables)>0:
            # X = X[:,~constant_variables]
            X_development = X_development[:,~constant_variables]
            X_heldout = X_heldout[:,~constant_variables]
            columns = columns[~constant_variables]
            print(f'Removed {np.sum(constant_variables)} columns')
        else:
            print('Not constant variables found ...')

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # FEATURE SELECTION (CHANING COLUMNS, NEED TO UDPATE ALL MATRICES)
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        print('Shapes of matrices before FS...')
        print(f'X_development: {X_development.shape}')
        print(f'y_development: {y_development.shape}')
        print(f'X_heldout:  {X_heldout.shape}')
        print(f'y_heldout:  {y_heldout.shape}')

        if 'feature_selection' in params and params['feature_selection']:
            print('Applying feature selection')
            clf = SelectKBest(f_classif, k=params['k_best_features'], ).fit(X_development, y_development)
            X_development = clf.transform(X_development)
            X_heldout = clf.transform(X_heldout)
            columns = clf.transform(columns.reshape(1,-1))[0,:]

        return X_development, y_development, X_heldout, y_heldout, columns
