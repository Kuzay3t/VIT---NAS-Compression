
# tests/test_search_space.py
"""
Unit tests for search space.
"""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search_space.search_space import SearchSpace, SearchConfig

class TestSearchSpace(unittest.TestCase):
    """Test SearchSpace class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.search_space = SearchSpace("config/search_space.yaml")
    
    def test_search_space_initialization(self):
        """Test that search space loads correctly."""
        self.assertIsNotNone(self.search_space.raw_config)
        self.assertIsNotNone(self.search_space.search_space_dict)
    
    def test_random_sampling(self):
        """Test random configuration sampling."""
        for _ in range(10):
            config = self.search_space.random_sample()
            self.assertIsInstance(config, SearchConfig)
            self.assertGreaterEqual(config.architecture.depth, 6)
            self.assertLessEqual(config.architecture.depth, 16)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = self.search_space.random_sample()
        is_valid, errors = self.search_space.validate_config(config)
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(errors, list)
    
    def test_config_to_dict(self):
        """Test conversion of config to dictionary."""
        config = self.search_space.random_sample()
        config_dict = self.search_space.to_dict(config)
        
        self.assertIn('architecture', config_dict)
        self.assertIn('compression', config_dict)
        self.assertIn('adaptivity', config_dict)
        self.assertIn('hardware', config_dict)
    
    def test_embed_dim_divisibility(self):
        """Test that embed_dim is divisible by num_heads."""
        for _ in range(50):
            config = self.search_space.random_sample()
            self.assertEqual(
                config.architecture.embed_dim % config.architecture.num_heads,
                0,
                f"embed_dim ({config.architecture.embed_dim}) not divisible by "
                f"num_heads ({config.architecture.num_heads})"
            )

if __name__ == "__main__":
    unittest.main()
