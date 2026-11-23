"""Abort of a prefill KV sender after the transfer worker has finished."""

import threading
import unittest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 7


def make_sender():
    manager = object.__new__(CommonKVManager)
    manager.request_status = {}
    manager.is_dummy_cp_rank = False
    manager.deferred_bootstrap = None
    manager._staging_outstanding = {}
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.defer_decode_allocation = False
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        sender = CommonKVSender(manager, "unused", ROOM)
    return sender, manager


class TestCommonSenderAbort(CustomTestCase):
    def test_abort_after_drained_success_keeps_success(self):
        sender, manager = make_sender()
        manager.request_status[ROOM] = KVPoll.Success

        sender.abort()

        self.assertEqual(manager.request_status[ROOM], KVPoll.Success)
        self.assertNotIn(ROOM, manager.failure_records)
        self.assertEqual(sender.conclude_state, KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_abort_with_outstanding_staging_chunk_fails(self):
        sender, manager = make_sender()
        manager.request_status[ROOM] = KVPoll.Success
        # The worker counted this chunk at dequeue and has not uncounted it.
        # poll() holds that Success as Transferring, same as the scheduler.
        manager._staging_outstanding[ROOM] = 1

        sender.abort()

        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager.failure_records[ROOM], "Aborted by AbortReq.")
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        # Failed stays the only absorbing state.
        manager.update_status(ROOM, KVPoll.Success)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)

    def test_abort_before_deferred_cleanup_still_fails(self):
        sender, manager = make_sender()
        manager.defer_decode_allocation = True
        manager.request_status[ROOM] = KVPoll.Success
        # Count is zero, but worker cleanup has not popped the room yet.
        # poll() keeps that as Transferring until the entry is gone.
        manager._staging_outstanding[ROOM] = 0

        sender.abort()

        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertIn(ROOM, manager.failure_records)
        self.assertEqual(sender.poll(), KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
