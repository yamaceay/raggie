from typing import List, Tuple, Dict
import os
import json
import csv

from .types import RaggieDataClass

class RaggieData(RaggieDataClass):
    """
    Default implementation of the RaggieDataClass.

    This class provides methods to load and manage training, testing, and
    validation data from a specified directory. It supports JSONL, JSON, and CSV
    file formats.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the RaggieData instance.

        Args:
            data_dir (str): Path to the directory containing data files.
        """
        self.data_dir = data_dir

        self.split_types = ['train', 'test', 'val']
        self.valid_file_types = ['.jsonl', '.json', '.csv']

        self.found_files = self.find_data_files()
        self.split_files = self.label_data_files()

        self.train_path = self.split_files.get('train')
        self.test_path = self.split_files.get('test')
        self.val_path = self.split_files.get('val')

        self._train_data = None
        self._test_data = None
        self._val_data = None

    @property
    def train_data(self) -> List[Tuple[str, str]]:
        """
        Load and return the training data.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing paired data.

        Raises:
            ValueError: If no training data file is found.
        """
        if not self.train_path:
            raise ValueError("No training data found. Please ensure a 'train' file is present in the data directory.")
        if self._train_data is not None:
            return self._train_data
        return self.load_data(self.train_path)
    
    @property
    def test_data(self) -> List[Tuple[str, str]]:
        """
        Load and return the testing data.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing paired data.

        Raises:
            ValueError: If no testing data file is found.
        """
        if not self.test_path:
            raise ValueError("No test data found. Please ensure a 'test' file is present in the data directory.")
        if self._test_data is not None:
            return self._test_data
        return self.load_data(self.test_path)
    
    @property
    def val_data(self) -> List[Tuple[str, str]]:
        """
        Load and return the validation data.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing paired data.

        Raises:
            ValueError: If no validation data file is found.
        """
        if not self.val_path:
            raise ValueError("No validation data found. Please ensure a 'val' file is present in the data directory.")
        if self._val_data is not None:
            return self._val_data
        return self.load_data(self.val_path)

    def find_data_files(self) -> List[str]:
        """
        Find and return a list of data files in the specified directory.

        Returns:
            List[str]: A list of file names found in the directory.
        """
        files = [f for f in os.listdir(self.data_dir) if f.endswith(tuple(self.valid_file_types))]
        return files
    
    def label_data_files(self) -> Dict[str, str]:
        """
        Label data files in the directory for training, testing, and validation.

        Returns:
            Dict[str, str]: A dictionary mapping data types to file paths.
        """
        labeled_files = {}
        for file in self.found_files:
            file_path = os.path.join(self.data_dir, file)
            for data_type in self.split_types:
                if data_type in file.lower():
                    labeled_files[data_type] = file_path
        return labeled_files

    def load_data(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Load data from the specified file.

        Args:
            file_path (str): Path to the data file.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing paired data.

        Raises:
            ValueError: If the file type is unsupported.
        """
        _, file_ext = os.path.splitext(file_path)
        if file_ext not in self.valid_file_types:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types are: {self.valid_file_types}")
        
        if file_ext == ".jsonl":
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    data_pair = (item["key"], item["value"])
                    data.append(data_pair)
        elif file_ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
                data = [(item["key"], item["value"]) for item in data]
        elif file_ext == ".csv":
            data = []
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data_pair = (row["key"], row["value"])
                    data.append(data_pair)
        return data