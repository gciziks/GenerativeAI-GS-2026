import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPipeline(unittest.TestCase):
    
    def test_imports(self):
        try:
            import data_generation
            import preprocessing
            import feature_engineering
            import train_models
            import evaluate_models
            import shap_explainer
            print("OK")
        except Exception as e:
            self.fail(f"Falha na importacao: {e}")
    
    def test_directories_exist(self):
        dirs = ['data', 'data/raw', 'data/processed', 'src', 'models', 
                'shap_outputs', 'app', 'tests', 'notebooks']
        for d in dirs:
            self.assertTrue(os.path.exists(d), f"Diretorio {d} nao existe")


if __name__ == '__main__':
    unittest.main()
