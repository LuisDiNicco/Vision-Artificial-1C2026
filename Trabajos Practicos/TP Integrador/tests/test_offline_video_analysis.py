import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tp_integrador.backend.celebrity import CelebrityIndex
from tp_integrador.backend.deteccion import FaceDetection
from tp_integrador.backend.offline_video_analysis import (
    OfflineVideoConfig,
    _Observation,
    _Quality,
    _Track,
    _TrackDecision,
    _decide_track,
    _fill_short_track_gaps,
    _frame_decisions_for_tracks,
    _measure_quality,
    _merge_compatible_tracks,
    _nms_detections,
    _track_score,
    _observations_from_feature_arrays,
    _observations_to_feature_arrays,
    _load_feature_cache,
    _save_feature_cache,
)


def detection(bbox):
    landmarks = np.zeros((478, 3), dtype=np.float32)
    return FaceDetection(bbox=bbox, landmarks=landmarks, confidence=1.0)


def quality():
    return _Quality(True, 0.8, "ok", 120, 40.0, 128.0, 80.0, 0.0)


class OfflineVideoAnalysisTests(unittest.TestCase):
    def test_blur_is_a_soft_penalty_offline(self):
        face = detection((10, 10, 130, 130))
        face.landmarks[[33, 133], :2] = (40, 40)
        face.landmarks[[362, 263], :2] = (80, 40)
        face.landmarks[1, :2] = (60, 55)
        face.landmarks[152, :2] = (60, 100)
        face.landmarks[61, :2] = (45, 80)
        face.landmarks[291, :2] = (75, 80)
        frame = np.full((160, 160, 3), 128, dtype=np.uint8)
        measured = _measure_quality(frame, face, OfflineVideoConfig())
        self.assertTrue(measured.usable)
        self.assertIn("desenfoque", measured.reason)
        self.assertGreater(measured.weight, 0.0)

    def test_nms_removes_duplicate_faces(self):
        detections = [
            detection((10, 10, 110, 110)),
            detection((12, 12, 108, 108)),
            detection((200, 20, 300, 120)),
        ]
        kept = _nms_detections(detections, threshold=0.68, limit=12)
        self.assertEqual(len(kept), 2)

    def test_track_decision_uses_temporal_votes_and_sample_support(self):
        person_a = np.zeros(512, dtype=np.float32)
        person_b = np.zeros(512, dtype=np.float32)
        person_a[0] = 1.0
        person_b[1] = 1.0
        samples = np.vstack([person_a, person_a, person_b]).astype(np.float32)
        index = CelebrityIndex(
            embeddings=np.vstack([person_a, person_b]),
            names=["Persona A", "Persona B"],
            image_paths=[Path("a.jpg"), Path("b.jpg")],
            counts=[2, 1],
            sample_embeddings=samples,
            sample_names=["Persona A", "Persona A", "Persona B"],
            sample_person_ids=np.array([0, 0, 1], dtype=np.int32),
        )
        track = _Track(0)
        for frame_index in range(5):
            embedding = person_a.copy()
            matches = index.top_unique(embedding, limit=3)
            track.observations.append(
                _Observation(
                    frame_index,
                    frame_index / 30.0,
                    detection((10, 10, 130, 130)),
                    quality(),
                    embedding,
                    matches,
                    track_id=0,
                )
            )
        decision = _decide_track(track, index, OfflineVideoConfig())
        self.assertEqual(decision.label, "Persona A")
        self.assertGreaterEqual(decision.votes, 3)
        self.assertGreaterEqual(decision.support_count, 2)

    def test_mixed_identity_track_is_rejected_conservatively(self):
        person_a = np.zeros(512, dtype=np.float32)
        person_b = np.zeros(512, dtype=np.float32)
        person_a[0] = 1.0
        person_b[1] = 1.0
        index = CelebrityIndex(
            embeddings=np.vstack([person_a, person_b]),
            names=["Persona A", "Persona B"],
            image_paths=[Path("a.jpg"), Path("b.jpg")],
            counts=[2, 2],
            sample_embeddings=np.vstack([person_a, person_a, person_b, person_b]),
            sample_names=["Persona A", "Persona A", "Persona B", "Persona B"],
            sample_person_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        )
        track = _Track(0)
        for frame_index, embedding in enumerate([person_a, person_a, person_a, person_b, person_b]):
            track.observations.append(
                _Observation(
                    frame_index,
                    frame_index / 30.0,
                    detection((10, 10, 130, 130)),
                    quality(),
                    embedding,
                    index.top_unique(embedding, limit=3),
                    track_id=0,
                )
            )

        decision = _decide_track(track, index, OfflineVideoConfig())

        self.assertEqual(decision.label, "desconocido")
        self.assertIn(
            decision.rejection_reason,
            {"votos_temporales_insuficientes", "identidades_competidoras"},
        )

    def test_bidirectional_confidence_varies_with_frame_evidence(self):
        person_a = np.zeros(512, dtype=np.float32)
        person_b = np.zeros(512, dtype=np.float32)
        person_a[0] = 1.0
        person_b[1] = 1.0
        index = CelebrityIndex(
            embeddings=np.vstack([person_a, person_b]),
            names=["Persona A", "Persona B"],
            image_paths=[Path("a.jpg"), Path("b.jpg")],
        )
        weaker = np.zeros(512, dtype=np.float32)
        weaker[0] = 0.48
        weaker[2] = np.sqrt(1.0 - weaker[0] ** 2)
        track = _Track(
            0,
            decision=_TrackDecision(label="Persona A", similarity=0.9, confidence=0.9),
        )
        for frame_index, embedding in enumerate([person_a, person_a, weaker, person_a, person_a]):
            track.observations.append(
                _Observation(
                    frame_index,
                    frame_index / 30.0,
                    detection((10, 10, 130, 130)),
                    quality(),
                    embedding,
                    index.top_unique(embedding, limit=3),
                    track_id=0,
                )
            )

        frame_decisions = _frame_decisions_for_tracks([track], OfflineVideoConfig())
        confidences = [frame_decisions[id(item)].confidence for item in track.observations]

        self.assertTrue(all(frame_decisions[id(item)].label == "Persona A" for item in track.observations))
        self.assertGreater(max(confidences) - min(confidences), 0.04)
        self.assertLess(confidences[2], confidences[1])

    def test_moderate_but_valid_evidence_does_not_collapse_to_unknown(self):
        person_a = np.zeros(512, dtype=np.float32)
        person_b = np.zeros(512, dtype=np.float32)
        person_a[0] = 1.0
        person_b[1] = 1.0
        index = CelebrityIndex(
            embeddings=np.vstack([person_a, person_b]),
            names=["Persona A", "Persona B"],
            image_paths=[Path("a.jpg"), Path("b.jpg")],
        )
        moderate = np.zeros(512, dtype=np.float32)
        moderate[0] = 0.48
        moderate[2] = np.sqrt(1.0 - moderate[0] ** 2)
        track = _Track(
            0,
            decision=_TrackDecision(label="Persona A", similarity=0.60, confidence=0.60),
        )
        for frame_index in range(40):
            track.observations.append(
                _Observation(
                    frame_index,
                    frame_index / 30.0,
                    detection((10, 10, 130, 130)),
                    quality(),
                    moderate.copy(),
                    index.top_unique(moderate, limit=3),
                    track_id=0,
                )
            )

        frame_decisions = _frame_decisions_for_tracks([track], OfflineVideoConfig())

        self.assertTrue(
            all(frame_decisions[id(item)].label == "Persona A" for item in track.observations)
        )

    def test_sustained_competing_identity_can_veto_a_segment(self):
        person_a = np.zeros(512, dtype=np.float32)
        person_b = np.zeros(512, dtype=np.float32)
        person_a[0] = 1.0
        person_b[1] = 1.0
        index = CelebrityIndex(
            embeddings=np.vstack([person_a, person_b]),
            names=["Persona A", "Persona B"],
            image_paths=[Path("a.jpg"), Path("b.jpg")],
        )
        track = _Track(
            0,
            decision=_TrackDecision(label="Persona A", similarity=0.90, confidence=0.90),
        )
        sequence = [person_a] * 12 + [person_b] * 12 + [person_a] * 12
        for frame_index, embedding in enumerate(sequence):
            track.observations.append(
                _Observation(
                    frame_index,
                    frame_index / 30.0,
                    detection((10, 10, 130, 130)),
                    quality(),
                    embedding,
                    index.top_unique(embedding, limit=3),
                    track_id=0,
                )
            )

        frame_decisions = _frame_decisions_for_tracks([track], OfflineVideoConfig())
        labels = [frame_decisions[id(item)].label for item in track.observations]

        self.assertEqual(labels[0], "Persona A")
        self.assertEqual(labels[-1], "Persona A")
        self.assertIn("desconocido", labels[12:24])

    def test_short_gap_is_interpolated_only_for_known_track(self):
        track = _Track(0, decision=_TrackDecision(label="Persona A", confidence=0.7))
        first = _Observation(0, 0.0, detection((10, 10, 110, 110)), quality(), None, track_id=0)
        last = _Observation(3, 0.1, detection((13, 10, 113, 110)), quality(), None, track_id=0)
        track.observations.extend([first, last])
        frames = [[first], [], [], [last]]
        count = _fill_short_track_gaps(frames, [track], OfflineVideoConfig(), fps=30.0)
        self.assertEqual(count, 2)
        self.assertTrue(frames[1][0].synthetic)
        self.assertTrue(frames[2][0].synthetic)

    def test_compatible_track_fragments_are_merged(self):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        left = _Track(0)
        right = _Track(1)
        left.observations.append(
            _Observation(2, 2 / 30, detection((10, 10, 110, 110)), quality(), embedding, track_id=0)
        )
        right.observations.append(
            _Observation(5, 5 / 30, detection((14, 10, 114, 110)), quality(), embedding, track_id=1)
        )
        merged = _merge_compatible_tracks([left, right], OfflineVideoConfig())
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].observations), 2)
        self.assertEqual(merged[0].observations[-1].track_id, 0)

    def test_same_position_does_not_join_different_identities(self):
        first_embedding = np.zeros(512, dtype=np.float32)
        second_embedding = np.zeros(512, dtype=np.float32)
        first_embedding[0] = 1.0
        second_embedding[1] = 1.0
        first = _Observation(
            0,
            0.0,
            detection((10, 10, 110, 110)),
            quality(),
            first_embedding,
        )
        second = _Observation(
            1,
            1 / 30,
            detection((10, 10, 110, 110)),
            quality(),
            second_embedding,
        )
        self.assertEqual(_track_score(first, second), 0.0)

    def test_feature_arrays_preserve_expensive_observation_data(self):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[7] = 1.0
        alternate = np.zeros(512, dtype=np.float32)
        alternate[9] = 1.0
        observation = _Observation(
            1,
            1 / 30,
            detection((10, 20, 130, 140)),
            quality(),
            embedding,
            scene_id=2,
            alternate_embedding=alternate,
        )
        arrays = _observations_to_feature_arrays([[], [observation]])
        restored = _observations_from_feature_arrays(arrays, total_frames=2, fps=30.0)[1][0]
        np.testing.assert_array_equal(restored.embedding, embedding)
        np.testing.assert_array_equal(restored.alternate_embedding, alternate)
        np.testing.assert_array_equal(restored.detection.landmarks, observation.detection.landmarks)
        self.assertEqual(restored.detection.bbox, observation.detection.bbox)
        self.assertEqual(restored.scene_id, 2)

    def test_feature_cache_round_trip_supports_current_numpy(self):
        arrays = {
            "frame_index": np.array([0, 2], dtype=np.int32),
            "embedding": np.eye(2, 512, dtype=np.float32),
        }
        metadata = {"version": 1, "frames": 3, "stats": {"detections": 2}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            _save_feature_cache(path, metadata, arrays)
            restored_metadata, restored_arrays = _load_feature_cache(path)

        self.assertEqual(restored_metadata, metadata)
        np.testing.assert_array_equal(restored_arrays["frame_index"], arrays["frame_index"])
        np.testing.assert_array_equal(restored_arrays["embedding"], arrays["embedding"])


if __name__ == "__main__":
    unittest.main()
