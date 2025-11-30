from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd


class DataGetter(ABC):
    """Abstract interface for data retrieval"""
    
    @abstractmethod
    def get_data(
        self, 
        cohorts: List[str], 
        schema: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve dataset for given cohorts.
        
        Args:
            cohorts: List of cohort identifiers
            schema: Optional list of feature IDs defining expected column order
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        pass