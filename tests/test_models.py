"""Exercise the extracted models on small, deterministic datasets."""

import os
import unittest

# Apply the notebook's macOS/OpenMP settings before numerical libraries load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification

from covidml.data import COVIDSequenceDataset, collate_sequences, make_sequence_loader
from covidml.evaluation import ModelResults
from covidml.models import (
    TCN,
    make_logistic_regression,
    make_xgboost_classifier,
    make_xgboost_pipeline,
    persistence_predictions,
)
from covidml.training import predict_tcn, train_tcn


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_persistence_uses_each_countrys_history(self):
        frame = pd.DataFrame(
            {
                "iso_code": ["A", "B", "A", "B", "A", "B"],
                "new_cases_smoothed_per_million": [100, 200, 125, 100, 124, 200],
            },
            index=[10, 20, 30, 40, 50, 60],
        )
        result = persistence_predictions(frame, lookback=1)
        np.testing.assert_array_equal(result, [0, 0, 1, 0, 0, 1])
        pd.testing.assert_index_equal(result.index, frame.index)

    def test_classical_models_fit_and_return_binary_probabilities(self):
        X, y = make_classification(
            n_samples=40, n_features=4, n_informative=3,
            n_redundant=0, random_state=42,
        )
        for model in (
            make_logistic_regression(C=0.1),
            make_xgboost_classifier(n_estimators=3, max_depth=2, n_jobs=1),
            make_xgboost_pipeline(n_estimators=3, max_depth=2, n_jobs=1),
        ):
            with self.subTest(model=type(model).__name__):
                model.fit(X, y)
                probabilities = model.predict_proba(X)
                self.assertEqual(probabilities.shape, (40, 2))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                self.assertTrue(np.isin(model.predict(X), [0, 1]).all())

    def test_tcn_cannot_see_future_timesteps(self):
        torch.manual_seed(42)
        model = TCN(2, 1, hidden_channels=4, num_layers=2, dropout=0).eval()
        features = torch.randn(1, 8, 2)
        static = torch.randn(1, 1)
        changed = features.clone()
        changed[:, 4:] += 100
        with torch.no_grad():
            before = model(features, static)
            after = model(changed, static)
        self.assertEqual(before.shape, (1, 8))
        torch.testing.assert_close(before[:, :4], after[:, :4])

    def test_sequences_preserve_country_groups_and_row_positions(self):
        raw = pd.DataFrame(
            {"iso_code": ["A", "B", "A"], "target_surge": [0, 1, 1]},
            index=[10, 40, 20],
        )
        features = pd.DataFrame(
            {"temporal": [1.0, 9.0, 2.0], "static": [3.0, 4.0, 3.0]},
        )
        original_index = features.index.copy()
        sequences = COVIDSequenceDataset(features, raw, ["temporal"], ["static"])
        np.testing.assert_array_equal(sequences[0][3], [0, 2])
        np.testing.assert_array_equal(sequences[1][3], [1])
        np.testing.assert_array_equal(sequences[0][0].numpy().ravel(), [1.0, 2.0])
        pd.testing.assert_index_equal(features.index, original_index)
        temps, stats, targets, _ = collate_sequences([sequences[0], sequences[1]])
        self.assertEqual(temps.shape, (2, 2, 1))
        self.assertEqual(stats.shape, (2, 1))
        self.assertEqual(targets.shape, (2, 2))
        self.assertEqual(temps[1, 1, 0].item(), 0)

    def test_tcn_training_and_prediction_restore_interleaved_rows(self):
        raw = pd.DataFrame(
            {"iso_code": ["A", "B", "A", "A"], "target_surge": [0, 1, 1, 0]},
            index=[10, 40, 20, 60],
        )
        features = pd.DataFrame(
            {"temporal": [0.1, 0.8, 0.4, 0.6], "static": [1.0, 2.0, 1.0, 1.0]}
        )
        dataset = COVIDSequenceDataset(features, raw, ["temporal"], ["static"])
        loader = make_sequence_loader(dataset, batch_size=2, shuffle=True)
        torch.manual_seed(42)
        model = TCN(1, 1, hidden_channels=4, num_layers=1, dropout=0)
        initial_weights = model.final_linear.weight.detach().clone()
        losses = train_tcn(model, loader, epochs=1)
        self.assertFalse(torch.equal(initial_weights, model.final_linear.weight))
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses).all())

        inference = COVIDSequenceDataset(
            features, raw.drop(columns="target_surge"), ["temporal"], ["static"]
        )
        probabilities = predict_tcn(model, make_sequence_loader(inference, batch_size=2))
        shuffled = predict_tcn(model, make_sequence_loader(inference, batch_size=1, shuffle=True))
        self.assertEqual(probabilities.shape, (4,))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities, shuffled, atol=1e-6)
        for country in ("A", "B"):
            mask = raw["iso_code"].eq(country).to_numpy()
            country_data = COVIDSequenceDataset(
                features.loc[mask], raw.loc[mask], ["temporal"], ["static"]
            )
            alone = predict_tcn(model, make_sequence_loader(country_data))
            np.testing.assert_allclose(probabilities[mask], alone, atol=1e-6)

    def test_tcn_training_ignores_padded_timesteps(self):
        raw = pd.DataFrame(
            {"iso_code": ["A", "A", "B"], "target_surge": [1, 0, 1]}
        )
        features = pd.DataFrame({"temporal": [1, 2, 3], "static": [1, 1, 2]})
        dataset = COVIDSequenceDataset(features, raw, ["temporal"], ["static"])
        model = TCN(1, 1, hidden_channels=2, num_layers=1, dropout=0)
        with torch.no_grad():
            model.final_linear.weight.zero_()
            model.final_linear.bias.zero_()
        losses = train_tcn(
            model, make_sequence_loader(dataset, batch_size=2),
            epochs=1, learning_rate=0, pos_weight=2,
        )
        # Two positive labels weigh twice as much as the one negative label.
        self.assertAlmostEqual(losses[0], 5 * np.log(2) / 3, places=6)

    def test_sequence_frames_must_have_corresponding_rows(self):
        raw = pd.DataFrame({"iso_code": ["A", "A"]})
        features = pd.DataFrame({"temporal": [1], "static": [1]})
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            COVIDSequenceDataset(features, raw, ["temporal"], ["static"])

    def test_evaluation_handles_no_predicted_surges(self):
        result = ModelResults.evaluate(
            "baseline", None, {}, [0, 1, 0, 1], [0, 0, 0, 0], [0.1, 0.4, 0.2, 0.3]
        )
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.precision, 0)
        self.assertEqual(result.f1, 0)
        self.assertEqual(result.roc_auc, 1)
        self.assertIn("baseline", str(result))
