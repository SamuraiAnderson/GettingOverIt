"""Offline guards against falsely accepting an ablation that did not execute."""
import unittest

from analyze_native_timing import validate_pose_sites


class PoseAblationChecks(unittest.TestCase):
    def sites(self):
        sites = [{"index": i, "calls": 0, "skippedCalls": 0, "operation": ""} for i in range(22)]
        sites[9].update(calls=3, skippedCalls=3, operation="hammerMeshHub.position")
        return sites

    def test_selected_write_must_actually_be_skipped(self):
        sites = self.sites()
        validate_pose_sites(sites, ["hammerMeshHub.position"])
        sites[9]["skippedCalls"] = 0
        with self.assertRaises(ValueError):
            validate_pose_sites(sites, ["hammerMeshHub.position"])

    def test_alias_cannot_silently_disable_ablation(self):
        sites = self.sites()
        sites[9].update(operation="center.position", skippedCalls=0)
        with self.assertRaises(ValueError):
            validate_pose_sites(sites, ["hammerMeshHub.position"])

    def test_unrequested_skip_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_pose_sites(self.sites(), [])

    def test_missing_statement_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_pose_sites(self.sites(), ["hammerMeshHub.position"], events=[])


if __name__ == "__main__":
    unittest.main()
