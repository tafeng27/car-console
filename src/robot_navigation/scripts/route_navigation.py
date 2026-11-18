import copy
import math

from navigation_geometry import point_heading
from navigation_plan import route_steps


def route_nodes(config, points, route):
    steps = route_steps(config, route)
    heading = math.radians(point_heading(config, points, config['start']))
    previous, after = config['start'], []
    for step in steps:
        if 'move' in step:
            goal, start = points[step['move']], points[previous]
            if step['mode'] in ('forward', 'reverse'):
                heading = math.atan2(goal['y'] - start['y'], goal['x'] - start['x'])
                if step['mode'] == 'reverse':
                    heading += math.pi
            previous = step['move']
        elif 'turn' in step:
            heading += math.radians(step['turn'])
        elif 'face' in step:
            heading = math.radians(step['face'])
        elif 'face_point' in step:
            heading = math.radians(point_heading(config, points, step['face_point']) + step.get('offset', 0))
        after.append(heading)
    entries = [-1] + [index for index, step in enumerate(steps) if 'move' in step]
    nodes = []
    for number, entry in enumerate(entries):
        end = entries[number + 1] if number + 1 < len(entries) else len(steps)
        name = config['start'] if entry == -1 else steps[entry]['move']
        entry_heading = math.radians(point_heading(config, points, config['start'])) if entry == -1 else after[entry]
        exit_heading = after[end - 1] if end > 0 else entry_heading
        before = [math.radians(point_heading(config, points, config['start']))] + after
        nodes.append({'name': name, 'begin': entry + 1, 'end': end,
                      'cursor_headings': {index: before[index] for index in range(entry + 1, end + 1)},
                      'entry_heading': entry_heading, 'exit_heading': exit_heading})
    return steps, nodes


def navigation_slice(config, points, route, source, target, cursor=None, heading=None):
    steps, nodes = route_nodes(config, points, route)
    names = {node['name'] for node in nodes}
    if source not in names or target not in names:
        raise ValueError('所选点位未加入路线%s，不能自动前往' % route)
    matching = [node for node in nodes if node['name'] == source]
    node = matching[0]
    known = cursor is not None and any(candidate['begin'] <= cursor <= candidate['end'] for candidate in matching)
    if known:
        node = next(candidate for candidate in matching if candidate['begin'] <= cursor <= candidate['end'])
        begin = cursor
    else:
        begin = node['begin']
    selected = []
    selected.append({'face': math.degrees(node['cursor_headings'][begin]), '_route_cursor_after': begin,
                     '_route_anchor': source})
    indexed = list(range(begin, len(steps))) + list(range(0, begin))
    found = target == source
    exit_index = node['end'] if found else None
    final_heading = node['exit_heading'] if found else None
    path = [source]
    for index in indexed:
        step = steps[index]
        if found and 'move' in step:
            break
        if 'move' in step:
            path.append(step['move'])
            if step['move'] == target:
                found = True
                target_node = next(value for value in nodes if value['begin'] == index + 1)
                exit_index, final_heading = target_node['end'], target_node['exit_heading']
        if 'task' not in step:
            selected.append({**step, '_route_cursor_after': (index + 1) % len(steps)})
    if not found:
        raise ValueError('路线中没有所选目标点位')
    if heading is not None and abs(math.atan2(math.sin(math.radians(heading) - final_heading), math.cos(math.radians(heading) - final_heading))) > math.radians(2):
        raise ValueError('单点导航的朝向需要按照路线执行')
    if heading is not None:
        final_heading = math.radians(heading)
    selected.append({'face': math.degrees(final_heading), '_route_cursor_after': exit_index % len(steps)})
    result = copy.deepcopy(config)
    result['start'] = source
    result['common'] = selected
    result['route1'], result['route2'], result['tail'] = [], [], []
    result['recognition']['enabled'] = False
    result['single_point_target'] = target
    result['single_point_final_heading'] = final_heading
    result['start_tolerance'] = 0.15
    return result, path
