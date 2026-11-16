import importlib
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch


analyze_module = importlib.import_module("allin1.analyze")


class ModelCacheTests(unittest.TestCase):
    def test_concurrent_requests_load_the_same_model_only_once(self):
        model = object()
        analyze_module._load_model_cached.cache_clear()
        try:
            with patch.object(
                analyze_module, "load_pretrained_model", return_value=model
            ) as load_model:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    models = list(
                        executor.map(
                            lambda _: analyze_module._get_model(
                                "harmonix-all", "cpu"
                            ),
                            range(2),
                        )
                    )

            self.assertEqual(load_model.call_count, 1)
            self.assertIs(models[0], model)
            self.assertIs(models[1], model)
        finally:
            analyze_module._load_model_cached.cache_clear()


if __name__ == "__main__":
    unittest.main()
