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
        Check if some of the attributes are None (Except the enums).
        This methods checks:
            - HCN code
            - CMG 
            - Case Weight and
            - Admit date
        Returns:
            True if any of those four attributes are None
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
        return ((field.name, getattr(self, field.name)) for field in fields(self))

    def __post_init__(self):
        if not self.admit_date is None:
            assert self.admit_date <= self.discharge_date

        if self.admit_category != AdmitCategory.NEW_BORN:
            assert 0<=self.age
        else: # NEW BORN
            assert -1<=self.age

    @staticmethod
    def from_dict_data(admit_id:int, admission:dict) -> Self:
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
        repr_ = f"<Admission Patient_code='{self.code}' "\
            f"admit='{self.admit_date.date()}' "\
                f"discharged='{self.discharge_date.date()}' "\
                    f"Age='{self.age}' gender='{self.gender}' ALC_days='{self.alc_days}' acute_days='{self.acute_days}' readmited=No>"
        if not self.readmission is None:
            repr_ = repr_[:-13] + f'readmited({self.readmission.admit_date.date()},{self.readmission.discharge_date.date()},{self.readmission.readmission_code})>'
        return repr_
    
    @staticmethod
    def diagnosis_codes_features(admissions: list[Self], min_df, vocabulary=None, use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix):
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
        """This methods take a list of admissions (Admission) and creates a dataframe with the diagnosis
        embeddings generated using Gensim "model_name" model.

        Args:
            admissions (list[Self]): List of admissions from which take the diagnosis to convert 
            into embeddings model_name (str, optional): Name of the already trained Gensim embedding 
            model for diagnoses. 

        Returns:
            pd.DataFrame: Returns a pandas dataframe with as many rows as elements in the admissions 
                          list and has many rows as the number of dimensions in the embedding model 
                          retrieved from disk.
        """
        print('Computing diagnosis embeddings ...')
        config = configuration.get_config()
        full_model_path = os.path.join(config['gensim_model_folder'], model_name)
        embedding_full_path = full_model_path+'_embeddings.npy'

        admission2embedding = {}
        # precomputed_found = 0
        if os.path.isfile(embedding_full_path) and not use_cached:
            print('Precomputed diagnosis model found but NOT BEING USED')
        if os.path.isfile(embedding_full_path) and use_cached:
            matrix = np.load(embedding_full_path)
            # If embedding_size=100, with Y number of admissions the matrix is (101, Y) shaped
            # The first column is the admit_id, the other 100 dimensions are the embeddings.
            admission2embedding = {admit_id: matrix[ix,1:] 
                                   for ix, admit_id in enumerate(matrix[:,0])}
            # precomputed_found = len(admission2embedding)   
        else:
            print('NOT USING CACHED MODEL (DIAGNOSIS)')
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
        """_summary_

        Args:
            admissions (list[Self]): _description_
            model_name (str, optional): _description_.

        Returns:
            pd.DataFrame: _description_
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
        return self.is_valid_testing_instance and not self.has_missing
    
    @property
    def is_valid_testing_instance(self:Self)->bool:
        return self.admit_category != AdmitCategory.CADAVER and \
                self.admit_category != AdmitCategory.STILLBORN and \
                not self.code is None

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # NUMERICAL FEATURES
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    @staticmethod
    def numerical_features(admissions: list[Self],) -> pd.DataFrame:
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
        config = configuration.get_config()
        return json.load(open(config['diagnosis_dict'], encoding='utf-8'))

    
    @staticmethod
    def get_intervention_mapping():
        config = configuration.get_config()
        return json.load(open(config['intervention_dict'], encoding='utf-8'))

    @staticmethod
    def get_y(admissions: list[Self])->np.ndarray:
        return np.array([1 if admission.has_readmission and \
                     admission.readmission.readmission_code!=ReadmissionCode.PLANNED_READMIT else 0 \
                     for admission in admissions])

    @property
    def has_readmission(self: Self,)->bool:
        return not self.readmission is None

    @property
    def length_of_stay(self: Self)->int:
        los = None
        if not self.admit_date is None:
            los = (self.discharge_date - self.admit_date).days
        return los
    
    def is_valid_readmission(self, readmission: Self)->bool:
        """
        Check if the readmission is valid. The readmission is valid if 
            - the readmission is at most 30 days later than the original admission (self) and
            - the readmission is as a readmission_core that indicates is a readmission and not a 
              first admission (or others).

        Args:
            readmission:The readmission to check if it is a valid readmission to the admission that receives the msg
        Returns:
            True if the `readmission` is a valid readmission for the admission that received the msg.
        """
        return (readmission.admit_date - self.discharge_date).days<=30 and \
            ReadmissionCode.is_readmit(readmission.readmission_code)
    
    def add_readmission(self, readmission: Self):
        """
        Adds the (re)admission sent as parameter as the readmission to the admission that receives the msg.
        Requires the readmission is valid. 

        Args:
            readmission: Admission to add as readmission.
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
        # print(set([str(admission.entry_code) for admission in all_admissions]))

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
        """_summary_

        Args:
            filtering (bool, optional): _description_. Defaults to True.
            combining_diagnoses (bool, optional): _description_. Defaults to False.
            combining_interventions (bool, optional): _description_. Defaults to False.

        Returns:
            list[Self]: _description_
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
        """_summary_

        Args:
            filtering (bool, optional): _description_. Defaults to True.
            combining_diagnoses (bool, optional): _description_. Defaults to False.
            combining_interventions (bool, optional): _description_. Defaults to False.

        Returns:
            list[Self]: _description_
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

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # FEATURE SELECTION (CHANING COLUMNS, NEED TO UDPATE ALL MATRICES)
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        io.debug('Shapes of matrices before FS...')
        io.debug(f'X: {X.shape}')
        io.debug(f'y: {y.shape}')



        return X, y, columns
    




    @staticmethod
    def get_development_and_held_out_matrices(params):
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
