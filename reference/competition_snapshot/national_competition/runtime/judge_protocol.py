"""Judge framing, one-connection round flow, and fresh-frame pan settling."""
import json
import math
import re
import socket
import struct
import time
from pathlib import Path

GOALS = ('CA001','CA002','CA003','CA004','CB001','CB002','CB003','CB004',
         'CC001','CC002','CC003','CC004','CD001','CD002','CD003','CD004')


def result_payload(results):
    counts = {}
    for table, report in results:
        if table not in (1,2,3) or report.get('status') != 'COMPLETE':
            raise ValueError('Only completed, correctly numbered tables can be submitted')
        for row in report.get('rows', []):
            cls, count = row['class_id'], row['count']
            if type(cls) is not int or not 0 <= cls < 16:
                raise ValueError('Invalid reportable class')
            if row['table'] != table or type(count) is not int or count < 1:
                raise ValueError('Invalid table/count')
            key = (table,GOALS[cls])
            if key in counts:
                raise ValueError('Duplicate table/goal')
            counts[key] = count
        for row in report.get('ocr', {}).get('results', []):
            goal = row.get('goal_id')
            if not goal:  # Preserve unmapped text in diagnostics; never invent Wxxx.
                continue
            if not re.fullmatch(r'W\d{3}',goal) or type(row['count']) is not int or row['count'] < 1:
                raise ValueError('Invalid official OCR code/count')
            key = (table,goal)
            if key in counts:
                raise ValueError('Duplicate OCR table/goal')
            counts[key] = row['count']
    lines = ['START'] + [f'Goal_ID={goal};Num={count};Table={table}'
                        for (table,goal),count in sorted(counts.items())] + ['END']
    return ('\r\n'.join(lines)+'\r\n').encode('ascii')


class JudgeConnection:
    def __init__(self, config, event):
        self.config, self.event = config, event
        self.sock = None
        self.types = []

    def begin(self):
        team = self.config['team_id']
        if not re.fullmatch(r'[A-Za-z0-9]+',team):
            raise ValueError('Team ID must contain letters/digits only')
        source = self.config.get('source_ip')
        self.sock = socket.create_connection((self.config['host'],self.config['port']),
            timeout=self.config.get('connect_timeout_seconds',1.5),
            source_address=(source,0) if source else None)
        self.sock.settimeout(2.)
        self.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        return self._send(0,team.encode('ascii'))

    def _send(self, kind, payload):
        # Detect a previously closed peer without waiting for an ACK (none exists).
        if self.types:
            import select
            if select.select([self.sock],[],[],0)[0] and self.sock.recv(1,socket.MSG_PEEK) == b'':
                raise ConnectionError('Judge disconnected; do not reconnect within this round')
        self.sock.sendall(struct.pack('>II',kind,len(payload))+payload)
        sent = time.monotonic()
        self.types.append(kind)
        self.event('packet_sent',data_type=kind,bytes=len(payload),sent_monotonic=sent)
        return sent

    def rotate(self):
        if self.types not in ([0],[0,3]):
            raise RuntimeError('Only two rotations are allowed after the start packet')
        return self._send(3,b'0000')

    def finish(self, payload, round_number):
        expected = [0] if round_number == 1 else [0,3,3]
        if self.types != expected:
            raise RuntimeError('Invalid round packet order')
        self._send(1,payload)
        # No retries: an ambiguous send could otherwise duplicate a rotation/result.

    def close(self):
        if self.sock:
            self.sock.close()


class StabilityGate:
    """Bounded pan wait; uncertain table geometry never aborts the round."""
    def __init__(self, rotation_sent, minimum_wait=2.5, stable_seconds=.8,
                 maximum_wait=7., relative_change=.035, require_stability=True):
        if maximum_wait < minimum_wait:
            raise ValueError('Pan wait limit must cover the minimum rotation wait')
        self.rotation_sent = rotation_sent
        self.minimum_wait, self.stable_seconds = minimum_wait, stable_seconds
        self.deadline = rotation_sent + maximum_wait
        self.relative_change = relative_change
        self.last_sequence = None
        self.previous = None
        self.anchor = None
        self.stable_since = None
        self.last_fresh = None
        self.frames = 0
        self.require_stability = require_stability
        self.release_reason = None
        self.stability_confirmed = False

    def reset(self):
        self.previous = None
        self.anchor = None
        self.stable_since = None
        self.frames = 0

    def observe(self, sequence, box, now, fresh=True):
        if self.release_reason is not None:
            return True
        if not self.require_stability and now-self.rotation_sent >= self.minimum_wait:
            self.release_reason = 'MINIMUM_WAIT_COMPLETE'
            return True
        if now >= self.deadline:
            self.release_reason = 'WAIT_LIMIT_CONTINUE'
            return True
        if not fresh or box is None:
            self.reset()
            return False
        if sequence == self.last_sequence:
            if self.last_fresh is not None and now-self.last_fresh > .4:
                self.reset()
            return False
        self.last_sequence = sequence
        if self.last_fresh is not None and now-self.last_fresh > .4:
            self.reset()
        self.last_fresh = now
        if now-self.rotation_sent < self.minimum_wait:
            self.reset()
            return False
        x1,y1,x2,y2 = box
        if not all(math.isfinite(x) for x in box) or x2 <= x1 or y2 <= y1:
            self.reset()
            return False
        geometry = ((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1)
        def change(reference):
            return max(abs(geometry[0]-reference[0])/reference[2],
                       abs(geometry[1]-reference[1])/reference[3],
                       abs(geometry[2]-reference[2])/reference[2],
                       abs(geometry[3]-reference[3])/reference[3])
        stable = (self.previous is not None and self.anchor is not None and
                  change(self.previous) <= self.relative_change and
                  change(self.anchor) <= self.relative_change)
        if not stable:
            self.stable_since, self.frames = now,0
            self.anchor=geometry
        self.previous = geometry
        self.frames += 1
        if self.frames >= 3 and now-self.stable_since >= self.stable_seconds:
            self.stability_confirmed = True
            self.release_reason = 'STABILITY_CONFIRMED'
            return True
        return False


def execute_round(connection, round_number, recognize, output, event):
    """Begin is deliberately outside: it runs before importing vision or opening camera."""
    completed = []
    for table in ((1,) if round_number == 1 else (1,2,3)):
        if table > 1 and connection is None:
            event('round_stopped_offline',next_table=table,completed_tables=[t for t,_ in completed],
                  output=str(output),reason='No judge connection; cannot control pan')
            return completed
        rotation_sent = connection.rotate() if table > 1 else None
        event('table_preparing',table=table,rotation_sent_monotonic=rotation_sent)
        report = recognize(table,rotation_sent)
        if report.get('status') != 'COMPLETE':
            raise RuntimeError(f'Table {table} did not complete')
        completed.append((table,report))
        payload = result_payload(completed)
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(output)
        event('table_complete',table=table,reason=report.get('reason'))
    if connection is None:
        event('round_saved_offline',completed_tables=[t for t,_ in completed],output=str(output))
        return completed
    connection.finish(payload,round_number)
    event('round_sent',packet_order=connection.types,output=str(output),
          note='sendall completed; judge protocol provides no receipt ACK')
    return completed
