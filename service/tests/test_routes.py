import unittest

from samm_server.routes import (
    embedding_job_id_from_path,
    embedding_job_items_id_from_path,
    finetuning_dataset_build_from_path,
    finetuning_dataset_from_path,
    finetuning_job_cancel_from_path,
    finetuning_job_from_path,
    finetuning_report_from_path,
    prediction_job_id_from_path,
    prediction_job_items_id_from_path,
    prepare_weight_id_from_path,
    video_prediction_job_cancel_id_from_path,
    video_prediction_job_frames_id_from_path,
    video_prediction_job_id_from_path,
    video_prediction_job_run_id_from_path,
    weight_id_from_path,
    worker_get_path,
    worker_post_path,
)
from samm_server.worker import video_prediction_cursor


class RouteTest(unittest.TestCase):
    def test_weight_id_from_path(self):
        self.assertEqual(weight_id_from_path("/models/sam_vit_b"), "sam_vit_b")

    def test_weight_id_from_path_rejects_collection_and_nested_paths(self):
        self.assertIsNone(weight_id_from_path("/models"))
        self.assertIsNone(weight_id_from_path("/models/"))
        self.assertIsNone(weight_id_from_path("/models/sam_vit_b/prepare"))

    def test_prepare_weight_id_from_path(self):
        self.assertEqual(prepare_weight_id_from_path("/models/sam_vit_b/prepare"), "sam_vit_b")

    def test_prepare_weight_id_from_path_rejects_other_paths(self):
        self.assertIsNone(prepare_weight_id_from_path("/models/sam_vit_b"))
        self.assertIsNone(prepare_weight_id_from_path("/models/sam_vit_b/prepare/again"))

    def test_worker_get_path(self):
        self.assertTrue(worker_get_path("/prepared"))
        self.assertTrue(worker_get_path("/embedding-jobs/job-1"))
        self.assertTrue(worker_get_path("/prediction-jobs/job-1"))
        self.assertTrue(worker_get_path("/video-prediction-jobs/job-1"))
        self.assertFalse(worker_get_path("/models"))
        self.assertFalse(worker_get_path("/models/sam_vit_b"))
        self.assertFalse(worker_get_path("/health"))

    def test_worker_post_path(self):
        self.assertTrue(worker_post_path("/embeddings"))
        self.assertTrue(worker_post_path("/embedding-files/load"))
        self.assertTrue(worker_post_path("/embedding-files/save"))
        self.assertTrue(worker_post_path("/embedding-jobs"))
        self.assertTrue(worker_post_path("/embedding-jobs/job-1/items"))
        self.assertTrue(worker_post_path("/prediction-jobs"))
        self.assertTrue(worker_post_path("/prediction-jobs/job-1/items"))
        self.assertTrue(worker_post_path("/video-prediction-jobs"))
        self.assertTrue(worker_post_path("/video-prediction-jobs/job-1/frames"))
        self.assertTrue(worker_post_path("/video-prediction-jobs/job-1/run"))
        self.assertTrue(worker_post_path("/video-prediction-jobs/job-1/cancel"))
        self.assertTrue(worker_post_path("/offload"))
        self.assertTrue(worker_post_path("/predict"))
        self.assertFalse(worker_post_path("/models/sam_vit_b/prepare"))
        self.assertFalse(worker_post_path("/models/sam_vit_b"))

    def test_embedding_job_id_from_path(self):
        self.assertEqual(embedding_job_id_from_path("/embedding-jobs/job-1"), "job-1")
        self.assertIsNone(embedding_job_id_from_path("/embedding-jobs"))
        self.assertIsNone(embedding_job_id_from_path("/embedding-jobs/job-1/nested"))

    def test_embedding_job_items_id_from_path(self):
        self.assertEqual(embedding_job_items_id_from_path("/embedding-jobs/job-1/items"), "job-1")
        self.assertIsNone(embedding_job_items_id_from_path("/embedding-jobs/job-1"))
        self.assertIsNone(embedding_job_items_id_from_path("/embedding-jobs/job-1/items/extra"))

    def test_prediction_job_id_from_path(self):
        self.assertEqual(prediction_job_id_from_path("/prediction-jobs/job-1"), "job-1")
        self.assertIsNone(prediction_job_id_from_path("/prediction-jobs"))
        self.assertIsNone(prediction_job_id_from_path("/prediction-jobs/job-1/nested"))

    def test_prediction_job_items_id_from_path(self):
        self.assertEqual(prediction_job_items_id_from_path("/prediction-jobs/job-1/items"), "job-1")
        self.assertIsNone(prediction_job_items_id_from_path("/prediction-jobs/job-1"))
        self.assertIsNone(prediction_job_items_id_from_path("/prediction-jobs/job-1/items/extra"))

    def test_video_prediction_job_paths(self):
        self.assertEqual(video_prediction_job_id_from_path("/video-prediction-jobs/job-1"), "job-1")
        self.assertEqual(video_prediction_job_frames_id_from_path("/video-prediction-jobs/job-1/frames"), "job-1")
        self.assertEqual(video_prediction_job_run_id_from_path("/video-prediction-jobs/job-1/run"), "job-1")
        self.assertEqual(video_prediction_job_cancel_id_from_path("/video-prediction-jobs/job-1/cancel"), "job-1")
        self.assertIsNone(video_prediction_job_id_from_path("/video-prediction-jobs/job-1/frames"))
        self.assertIsNone(video_prediction_job_frames_id_from_path("/video-prediction-jobs/job-1/frames/extra"))

    def test_video_prediction_cursor(self):
        self.assertEqual(video_prediction_cursor(""), 0)
        self.assertEqual(video_prediction_cursor("cursor=0"), 0)
        self.assertEqual(video_prediction_cursor("cursor=12"), 12)
        self.assertIsNone(video_prediction_cursor("cursor=-1"))
        self.assertIsNone(video_prediction_cursor("cursor="))
        self.assertIsNone(video_prediction_cursor("cursor=1&cursor=2"))

    def test_finetuning_dataset_from_path(self):
        self.assertEqual(finetuning_dataset_from_path("/finetuning/datasets/tumor"), "tumor")
        self.assertIsNone(finetuning_dataset_from_path("/finetuning/datasets"))
        self.assertIsNone(finetuning_dataset_from_path("/finetuning/datasets/tumor/build"))

    def test_finetuning_dataset_build_from_path(self):
        self.assertEqual(finetuning_dataset_build_from_path("/finetuning/datasets/tumor/build"), "tumor")
        self.assertIsNone(finetuning_dataset_build_from_path("/finetuning/datasets/tumor"))
        self.assertIsNone(finetuning_dataset_build_from_path("/finetuning/datasets/tumor/build/extra"))

    def test_finetuning_report_from_path(self):
        self.assertEqual(finetuning_report_from_path("/finetuning/reports/tumor_v1"), "tumor_v1")
        self.assertIsNone(finetuning_report_from_path("/finetuning/reports"))
        self.assertIsNone(finetuning_report_from_path("/finetuning/reports/tumor_v1/extra"))

    def test_finetuning_job_from_path(self):
        self.assertEqual(finetuning_job_from_path("/finetuning/jobs/job-1"), "job-1")
        self.assertIsNone(finetuning_job_from_path("/finetuning/jobs"))
        self.assertIsNone(finetuning_job_from_path("/finetuning/jobs/job-1/cancel"))

    def test_finetuning_job_cancel_from_path(self):
        self.assertEqual(finetuning_job_cancel_from_path("/finetuning/jobs/job-1/cancel"), "job-1")
        self.assertIsNone(finetuning_job_cancel_from_path("/finetuning/jobs/job-1"))
        self.assertIsNone(finetuning_job_cancel_from_path("/finetuning/jobs/job-1/cancel/extra"))


if __name__ == "__main__":
    unittest.main()
