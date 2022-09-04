"""Database class to handle models and sequences"""

import logging

logger = logging.getLogger("zm_mlapi")

class Database:

    def __init__(self, settings):
        self.settings = settings
        self.db = self.settings.db
        self.db_path = self.settings.db_path
        self.models = self.settings.models
        self.sequences = self.settings.sequences
        self.db.create_tables([self.models, self.sequences])
        self.available_models = self.models.select().dicts()
        self.available_sequences = self.sequences.select().dicts()
        self.available_model_names = [model["name"] for model in self.available_models]
        self.available_sequence_names = [sequence["name"] for sequence in self.available_sequences]
        self.available_model_paths = [model["path"] for model in self.available_models]
        self.available_sequence_paths = [sequence["path"] for sequence in self.available_sequences]
        self.available_model_ids = [model["id"] for model in self.available_models]
        self.available_sequence_ids = [sequence["id"] for sequence in self.available_sequences]
        self.available_model_names = [model["name"] for model in self.available_models]
        self.available_sequence_names = [sequence["name"] for sequence in self.available_sequences]
        self.available_model_paths = [model["path"] for model in self.available_models]
        self.available_sequence_paths = [sequence["path"] for sequence in self.available_sequences]
        self.available_model_ids = [model["id"] for model in self.available_models]
        self.available_sequence_ids = [sequence["id"] for sequence in self.available_sequences]

    def add_model(self, name, path, description):
        if name in self.available_model_names:
            logger.warning(f"model '{name}' already exists")
            return
        else:
            self.models.insert(name=name, path=path, description=description).execute()
            self.available_models = self.models.select().dicts()
            self.available_model_names = [model["name"] for model in self.available_models]
            self.available_model_paths = [model["path"] for model in self.available_models]
            self.available_model_ids = [model["id"] for model in self.available_models]
            logger.info(f"model '{name}' added to database")

    def add_sequence(self, name, path, description):
        if name in self.available_sequence_names:
            logger.warning(f"sequence '{name}' already exists")
            return
        else:
            self.sequences.insert(name=name, path=path, description=description).execute()
            self.available_sequences
