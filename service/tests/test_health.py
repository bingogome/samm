import unittest

from samm_server.protocol import PROTOCOL_VERSION, SERVICE_NAME, SERVICE_VERSION, health_payload, offload_payload, prepared_payload


class HealthPayloadTest(unittest.TestCase):
    def test_health_payload(self):
        self.assertEqual(
            health_payload(),
            {
                "name": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "status": "ready",
                "protocol_version": PROTOCOL_VERSION,
            },
        )

    def test_empty_prepared_payload(self):
        self.assertEqual(prepared_payload(None), {"prepared": None})

    def test_offload_payload(self):
        self.assertEqual(offload_payload(), {"status": "offloaded"})


if __name__ == "__main__":
    unittest.main()
