from typing import List, Dict, Tuple
import os
import json
import csv

from .types import RaggieDataClass, RaggieDataLoaderClass

valid_file_types = ['.jsonl', '.json', '.csv']

class RaggieData(RaggieDataClass):
    """
    Abstract base class for Raggie data handling.

    This class defines the interface for iterating over paired data.
    Users can implement their own data handling logic by extending this class.
    """

    def __init__(self, file_path: str):
        """
        Initialize the RaggieData instance.
        Args:
            file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.valid_file_types = valid_file_types
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        if not os.path.isfile(self.file_path):
            raise ValueError(f"The path {self.file_path} is not a file.")

        _, self.file_ext = os.path.splitext(self.file_path)
        if self.file_ext not in self.valid_file_types:
            raise ValueError(f"Unsupported file type: {self.file_ext}. Supported types are: {self.valid_file_types}")

        self._data = []

    @property
    def data(self) -> List[Tuple[str, str]]:
        """
        Get the paired data.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing paired data.
        """
        if not self._data:
            self._data = []
            if self.file_ext == ".jsonl":
                with open(self.file_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        data_pair = (item["key"], item["value"])
                        self._data.append(data_pair)
            elif self.file_ext == ".json":
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    data = [(item["key"], item["value"]) for item in data]
                    self._data.extend(data)
            elif self.file_ext == ".csv":
                with open(self.file_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data_pair = (row["key"], row["value"])
                        self._data.append(data_pair)
        return self._data
    

class RaggieDataLoader(RaggieDataLoaderClass):
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
        self.valid_file_types = valid_file_types

        self.found_files = self.find_data_files()
        self.split_files = self.label_data_files()

        self.train_path = self.split_files.get('train')
        self.test_path = self.split_files.get('test')
        self.val_path = self.split_files.get('val')

        self._train_data = None
        self._test_data = None
        self._val_data = None

    @property
    def train(self) -> RaggieDataClass:
        """
        Load and return the training data.

        Returns:
            RaggieDataClass: An instance of RaggieDataClass containing the training data.

        Raises:
            ValueError: If no training data file is found.
        """
        if not self.train_path:
            raise ValueError("No training data found. Please ensure a 'train' file is present in the data directory.")
        if self._train_data is None:
            self._train_data = RaggieData(self.train_path)
        return self._train_data
    
    @property
    def test(self) -> RaggieDataClass:
        """
        Load and return the testing data.

        Returns:
            RaggieDataClass: An instance of RaggieDataClass containing the testing data.

        Raises:
            ValueError: If no testing data file is found.
        """
        if not self.test_path:
            raise ValueError("No test data found. Please ensure a 'test' file is present in the data directory.")
        if self._test_data is None:
            self._test_data = RaggieData(self.test_path)
        return self._test_data

    @property
    def val(self) -> RaggieDataClass:
        """
        Load and return the validation data.

        Returns:
            RaggieDataClass: An instance of RaggieDataClass containing the validation data.

        Raises:
            ValueError: If no validation data file is found.
        """
        if not self.val_path:
            raise ValueError("No validation data found. Please ensure a 'val' file is present in the data directory.")
        if self._val_data is None:
            self._val_data = RaggieData(self.val_path)
        return self._val_data

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