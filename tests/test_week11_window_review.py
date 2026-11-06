"""Check that fixture exclusion and future-window boundaries stay explicit."""
import numpy as np
from review_week11_event_windows import contacts_by_frame, first_after, window_summary


def test_contact_scope_keeps_fixture_evidence_and_ignores_self_contact():
    def geom(i, kind, obj=None):
        return {'geom_id': i, 'class': kind, 'object_id': obj, 'name': f'g{i}',
                'collision_enabled': True, 'finger_sides': ['left_finger'] if i == 0 else []}
    g = {'geoms': [geom(0, 'gripper'), geom(1, 'movable_object', 'cup'),
                   geom(2, 'excluded_fixed_object', 'drawer'), geom(3, 'gripper')],
         'objects': [{'object_id': 'cup', 'movable_free_object': True},
                     {'object_id': 'drawer', 'movable_free_object': False}]}
    c = [{'geom1': 0, 'geom2': i, 'distance_m': -.001, 'efc_address': 0} for i in (1, 2, 3)]
    rows = contacts_by_frame(g, [{'frame_index': 0, 'contacts': c}])
    assert rows[0]['movable_objects'] == ['cup']
    assert rows[0]['excluded_counterparts'] == [['g2', 'excluded_fixed_object', 'drawer']]
    assert rows[0]['finger_sides']['cup'] == ['left_finger']
    for pair in c:
        pair['efc_address'] = -1
    row = contacts_by_frame(g, [{'frame_index': 0, 'contacts': c}])[0]
    assert not row['movable_objects'] and not row['excluded_counterparts']


def test_first_event_requires_new_onset():
    events = [{'start': s, 'left_censored': False, 'event_type': 'contact'} for s in (8, 10, 33)]
    assert first_after(events, 10)['start'] == 33
    assert first_after(events, 33) is None


def test_contact_at_anchor_is_not_reported_as_future_contact():
    c = {'frames': 35, 'events': [], 'ambiguities': []}
    g = {'objects': [{'object_id': 'cup'}]}
    ev = {'actions': np.full((35, 7), -1.), 'state': np.zeros((35, 8)),
          'object_position': np.zeros((35, 1, 3))}
    rows = [{'frame_index': i, 'movable_objects': ['cup'] if i == 0 else [],
             'excluded_counterparts': []} for i in range(35)]
    w = window_summary(c, g, ev, rows, 0)
    assert w['first_movable_contact_offset'] is None
    assert w['dense_contacts'][0]['movable_objects'] == ['cup']
    rows[32]['movable_objects'] = ['cup']
    assert window_summary(c, g, ev, rows, 0)['first_movable_contact_offset'] == 32
