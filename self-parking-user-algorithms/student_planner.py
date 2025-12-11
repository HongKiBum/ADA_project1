from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import heapq
import numpy as np


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ------------------------------
# 차량 외곽 좌표 계산 (충돌 여유 반영)
# ------------------------------
def get_vehicle_corners(x, y, yaw, length=4.8, width=2.2):
    # 실제 차체보다 약간 키운 길이/너비 → 안전 마진
    half_l = length / 2.0 + 0.2
    half_w = width / 2.0 + 0.1
    corners = [
        (half_l,  half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l,  half_w)
    ]
    c, s = math.cos(yaw), math.sin(yaw)
    world_corners = []
    for cx, cy in corners:
        wx = x + (cx * c - cy * s)
        wy = y + (cx * s + cy * c)
        world_corners.append((wx, wy))
    return world_corners


# ======================================================================
#                         주차 경로 플래너 본체
#  - 거리맵 기반 코스트맵 생성
#  - 3단계 Fail-Safe A* (Constraint / Standard / Relaxed)
#  - 경로 스무딩 + 슬롯 직선 진입
#  - Pure-Pursuit 제어
# ======================================================================
@dataclass
class PlannerSkeleton:
    # 맵/격자 정보
    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None  # (xmin, xmax, ymin, ymax)
    cell_size: float = 0.5

    grid_width: int = 0
    grid_height: int = 0

    # 코스트맵 (기본 / 타겟 반영)
    base_cost_map: Optional[np.ndarray] = None
    cost_map: Optional[np.ndarray] = None

    # 장애물 정보 (주차 슬롯, 라인)
    slots: List[Tuple[float, float, float, float]] = None
    lines: List[Tuple[float, float, float, float]] = None

    # 최종 경로 (웨이포인트 리스트)
    waypoints: List[Tuple[float, float]] = None

    # 타겟 슬롯 / 진입 정보
    target_slot: Optional[List[float]] = None
    target_center: Optional[Tuple[float, float]] = None
    target_entry_vec: Optional[Tuple[float, float]] = None  # 슬롯으로 들어가는 방향 (dx, dy)

    planning_done: bool = False
    debug_logged: bool = False
    frame_counter: int = 0
    viz_stride: int = 50
    viz_enabled: bool = False
    is_parked: bool = False

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        if self.slots is None:
            self.slots = []
        if self.lines is None:
            self.lines = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        self.map_data = map_payload
        self.map_extent = tuple(map(float, map_payload.get("extent", (0, 0, 0, 0))))
        self.cell_size = float(map_payload.get("cellSize", 0.5))

        raw_slots = map_payload.get("slots") or []
        self.slots = [tuple(map(float, s)) for s in raw_slots]

        raw_lines = map_payload.get("lines") or []
        self.lines = [tuple(map(float, ln)) for ln in raw_lines]

        self.waypoints.clear()
        self.planning_done = False
        self.is_parked = False

        self._precompute_base_map()

    # 월드 좌표 → 그리드 좌표 변환
    def _to_grid(self, wx, wy):
        if not self.map_extent:
            return 0, 0
        xmin, xmax, ymin, ymax = self.map_extent
        gx = int((wx - xmin) / self.cell_size)
        gy = int((wy - ymin) / self.cell_size)
        gx = max(0, min(self.grid_width - 1, gx))
        gy = max(0, min(self.grid_height - 1, gy))
        return gx, gy

    # 순수 파이썬 거리맵 → 코스트맵 생성
    #  장애물에서의 거리 계산 (8방향 다익스트라)
    #  차량 반경/안전 마진 반영해서 코스트로 변환
    def _precompute_base_map(self):
        """슬롯/라인/테두리를 장애물로 보고 거리맵 + 코스트맵 생성"""
        if not self.map_extent:
            return
        xmin, xmax, ymin, ymax = self.map_extent

        # 격자 크기 계산
        width_m = xmax - xmin
        height_m = ymax - ymin
        self.grid_width = int(np.ceil(width_m / self.cell_size))
        self.grid_height = int(np.ceil(height_m / self.cell_size))

        # 1. 장애물 초기화 (1.0 = 장애물 셀)
        base_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # (1-1) 슬롯 영역을 장애물로 마킹 (약간 인플레이션)
        for s in self.slots:
            gx1, gy1 = self._to_grid(s[0], s[2])
            gx2, gy2 = self._to_grid(s[1], s[3])
            r_min, r_max = min(gy1, gy2), max(gy1, gy2)
            c_min, c_max = min(gx1, gx2), max(gx1, gx2)
            r_min = max(0, r_min - 1)
            r_max = min(self.grid_height, r_max + 1)
            c_min = max(0, c_min - 1)
            c_max = min(self.grid_width, c_max + 1)
            base_map[r_min:r_max, c_min:c_max] = 1.0

        # (1-2) 주차장 라인(벽 등)을 장애물로 마킹
        for (x1, y1, x2, y2) in self.lines:
            dist = math.hypot(x2 - x1, y2 - y1)
            steps = int(dist / (self.cell_size * 0.5)) + 1
            for s in range(steps):
                t = s / steps
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                gx, gy = self._to_grid(px, py)
                base_map[gy, gx] = 1.0

        # (1-3) 맵 테두리를 전부 장애물로 처리 (벽)
        base_map[0, :] = 1
        base_map[-1, :] = 1
        base_map[:, 0] = 1
        base_map[:, -1] = 1

        # 2. 8방향 다익스트라로 거리맵(Distance Field) 계산
        H, W = self.grid_height, self.grid_width
        dist_map = np.full((H, W), np.inf, dtype=np.float32)
        pq = []

        # 장애물 위치 → 거리 0으로 초기화
        obs_y, obs_x = np.where(base_map > 0.5)
        for y, x in zip(obs_y, obs_x):
            dist_map[y, x] = 0.0
            heapq.heappush(pq, (0.0, x, y))

        # 8방향 인접 (상하좌우 + 대각선)
        dirs = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414),
            (-1, 1, 1.414), (-1, -1, 1.414),
        ]

        while pq:
            d, cx, cy = heapq.heappop(pq)

            if d > dist_map[cy, cx]:
                continue

            # 너무 멀리(> 약 8칸)까지 거리 계산은 생략
            if d > 8.0:
                continue

            for dx, dy, w in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    new_dist = d + w
                    if new_dist < dist_map[ny, nx]:
                        dist_map[ny, nx] = new_dist
                        heapq.heappush(pq, (new_dist, nx, ny))

        # 3. 거리맵 → 코스트맵 변환
        vehicle_radius_px = 3.0  # 0.5m * 3 = 1.5m
        final_map = np.ones_like(base_map, dtype=np.float32)

        safe_mask = dist_map > vehicle_radius_px
        final_map[safe_mask] = 1.0 - (dist_map[safe_mask] - vehicle_radius_px) / 20.0

        final_map[safe_mask] = np.clip(final_map[safe_mask], 0.05, 0.9)

        self.base_cost_map = final_map
        print("[Algo] Custom Distance Map built successfully.")

    def _prepare_cost_map_for_target(self, target_idx):
        if self.base_cost_map is None:
            self._precompute_base_map()
        self.cost_map = self.base_cost_map.copy()
        if target_idx != -1:
            self._carve_target_path(target_idx)

    def _carve_target_path(self, target_idx):
        slot = self.slots[target_idx]
        sx1, sx2, sy1, sy2 = slot
        cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        dx, dy = self.target_entry_vec

        carve_dist = 6.0  # 슬롯 앞에 약 6m 정도 정렬 공간 확보
        steps = int(carve_dist / (self.cell_size * 0.5))
        carve_width = 2

        for s in range(steps):
            d = s * (self.cell_size * 0.5)
            wx = cx + dx * d
            wy = cy + dy * d
            gx, gy = self._to_grid(wx, wy)
            for py in range(-carve_width, carve_width + 1):
                for px in range(-carve_width, carve_width + 1):
                    nx, ny = gx + px, gy + py
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.cost_map[ny, nx] = 0.01


    def _is_node_behind_target(self, nx, ny, xmin, ymin):
        if self.target_center is None or self.target_entry_vec is None:
            return False

        wx = xmin + (nx + 0.5) * self.cell_size
        wy = ymin + (ny + 0.5) * self.cell_size

        tcx, tcy = self.target_center
        dx, dy = self.target_entry_vec

        boundary_x = tcx + dx * 2.0
        boundary_y = tcy + dy * 2.0

        if (wx - boundary_x) * dx + (wy - boundary_y) * dy > 0:
            return True
        return False

    # 3단계 Fail-Safe A* (Constraint / Standard / Relaxed)
    def generate_path_astar(self, start_pos, target_pos, use_constraint=True, max_cost=0.6):
        """
        A* 경로 탐색.
        """
        xmin, _, ymin, _ = self.map_extent
        start_node = self._to_grid(*start_pos)
        goal_node = self._to_grid(*target_pos)

        def to_world(g):
            return xmin + (g[0] + 0.5) * self.cell_size, ymin + (g[1] + 0.5) * self.cell_size

        # 목표 지점이 거의 장애물인 경우 → 경로 없음
        if self.cost_map[goal_node[1], goal_node[0]] > 0.95:
            return []

        # 시작 지점이 고비용 지역일 경우 약간 완화 (출발은 할 수 있게)
        if self.cost_map[start_node[1], start_node[0]] > max_cost:
            self.cost_map[start_node[1], start_node[0]] = 0.1

        open_set = []
        heapq.heappush(open_set, (0, 0, start_node))
        came_from = {}
        g_score = {start_node: 0}

        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        final_node = None
        iterations = 0
        max_iterations = 6000

        best_dist = float('inf')
        best_node = start_node

        while open_set:
            iterations += 1
            if iterations > max_iterations:
                # 시간 제한 초과 시, 지금까지 목표에 가장 가까운 노드를 사용
                final_node = best_node
                break

            _, current_cost, current = heapq.heappop(open_set)

            d_goal = math.hypot(current[0] - goal_node[0], current[1] - goal_node[1])
            if d_goal < best_dist:
                best_dist = d_goal
                best_node = current

            # 목표 근처(그리드 거리 < 2)면 도착으로 간주
            if d_goal < 2.0:
                final_node = current
                break

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue

                # 코스트 상한(0.6 / 0.85) 초과 영역은 탐색하지 않음
                if self.cost_map[ny, nx] > max_cost:
                    continue

                # Constraint 단계에서는 슬롯 뒤쪽 영역 탐색 금지
                if use_constraint and self._is_node_behind_target(nx, ny, xmin, ymin):
                    continue

                field_cost = self.cost_map[ny, nx] * 10.0
                move_cost = math.hypot(dx, dy) + field_cost
                new_g = g_score[current] + move_cost

                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    h = math.hypot(nx - goal_node[0], ny - goal_node[1]) * 1.5
                    heapq.heappush(open_set, (new_g + h, new_g, (nx, ny)))
                    came_from[(nx, ny)] = current

        if not final_node:
            return []

        # 경로 복원
        path = []
        curr = final_node
        while curr in came_from:
            path.append(to_world(curr))
            curr = came_from[curr]
        path.append(to_world(start_node))
        path.reverse()

        # A* 결과를 그대로 쓰지 않고, 스무딩 후 사용
        return self._smooth_path_advanced(path)

    # Line-of-Sight 기반 경로 스무딩
    def _smooth_path_advanced(self, raw_path):
        """
        A* 경로는 계단 모양이므로,
        - 촘촘히 샘플링(dense_path)
        - Bresenham(Line-of-Sight) 체크로 직선으로 연결 가능한 구간만 남김
        → 차량이 따라가기 좋은 부드러운 경로로 변환
        """
        if len(raw_path) < 3:
            return raw_path

        dense_path = []
        for i in range(len(raw_path) - 1):
            p1, p2 = raw_path[i], raw_path[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            num_points = int(dist / 0.5) + 1
            for j in range(num_points):
                alpha = j / num_points
                mx = p1[0] + alpha * (p2[0] - p1[0])
                my = p1[1] + alpha * (p2[1] - p1[1])
                dense_path.append((mx, my))
        dense_path.append(raw_path[-1])

        tightened = [dense_path[0]]
        i = 1
        while i < len(dense_path):
            # 현 점까지 직선 LOS가 끊기면, 바로 이전 점을 웨이포인트로 추가
            if not self._check_line_of_sight_strict(tightened[-1], dense_path[i]):
                tightened.append(dense_path[i - 1])
            if i == len(dense_path) - 1:
                tightened.append(dense_path[i])
            i += 1
        return tightened

    def _check_line_of_sight_strict(self, p1, p2):
        r1, c1 = self._to_grid(p1[0], p1[1])
        r2, c2 = self._to_grid(p2[0], p2[1])
        if not self._bresenham_check(r1, c1, r2, c2, threshold=0.7):
            return False
        return True

    def _bresenham_check(self, r1, c1, r2, c2, threshold):
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc

        curr_r, curr_c = r1, c1
        while True:
            if not (0 <= curr_r < self.grid_height and 0 <= curr_c < self.grid_width):
                return False
            if self.cost_map[curr_r, curr_c] > threshold:
                return False
            if curr_r == r2 and curr_c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                curr_r += sr
            if e2 < dr:
                err += dr
                curr_c += sc
        return True

    # 전체 플래닝: 타겟 슬롯/게이트/3단계 A* + 직선 진입
    def compute_planning(self, obs):
        """
        한 번만 실행되는 메인 플래닝 함수.
        - 시작 위치(sx, sy)
        - 타겟 슬롯 중심/크기
        - 슬롯 진입 방향(target_entry_vec)
        - 게이트 포인트 탐색
        - 3단계 A* (Constraint → Standard → Relaxed)
        - 초반 직선화 + 마지막 슬롯 직선 진입
        → 최종 waypoints 생성
        """
        state = obs.get("state", {})
        sx, sy = float(state.get("x", 0)), float(state.get("y", 0))
        target = obs.get("target_slot")
        if not target:
            return

        # 타겟 슬롯 중심/사이즈
        t_cx = (target[0] + target[1]) / 2.0
        t_cy = (target[2] + target[3]) / 2.0
        w, h = target[1] - target[0], target[3] - target[2]

        # 맵 중심
        map_cx = (self.map_extent[0] + self.map_extent[1]) / 2.0
        map_cy = (self.map_extent[2] + self.map_extent[3]) / 2.0

        # 슬롯이 세로형이면 위/아래로, 가로형이면 좌/우로 진입 방향 결정
        dx, dy = 0.0, 0.0
        if h > w:
            dy = -1.0 if t_cy > map_cy else 1.0
        else:
            dx = -1.0 if t_cx > map_cx else 1.0

        self.target_center = (t_cx, t_cy)
        self.target_entry_vec = (dx, dy)

        # 가장 가까운 슬롯 인덱스 찾기 (실제 타겟 슬롯 인근 슬롯 선택)
        best_idx = -1
        min_d = float('inf')
        for i, s in enumerate(self.slots):
            cx = (s[0] + s[1]) / 2
            cy = (s[2] + s[3]) / 2
            d = (t_cx - cx) ** 2 + (t_cy - cy) ** 2
            if d < min_d:
                min_d, best_idx = d, i

        print(f"[Decision] Planning for Target #{best_idx}")
        self._prepare_cost_map_for_target(best_idx)

        # ----------------------------
        # 게이트 포인트 탐색
        #  - 슬롯 중심에서 진입 방향으로 나가면서
        #    코스트가 낮은 지점을 게이트로 선택
        # ----------------------------
        gate_dist = 5.0
        gx, gy = 0, 0
        found_safe_gate = False

        while gate_dist >= 1.5:
            gx = t_cx + dx * gate_dist
            gy = t_cy + dy * gate_dist
            g_grid_x, g_grid_y = self._to_grid(gx, gy)

            if not (0 <= g_grid_x < self.grid_width and 0 <= g_grid_y < self.grid_height):
                gate_dist -= 0.5
                continue

            if self.cost_map[g_grid_y, g_grid_x] < 0.6:
                found_safe_gate = True
                break
            gate_dist -= 0.5

        # 게이트를 못 찾으면 → 슬롯 중심으로 직접 진입 (fallback)
        if not found_safe_gate:
            print("[Warning] No safe gate! Using slot center.")
            gx, gy = t_cx, t_cy

        print(f"[Decision] Gate: ({gx:.1f}, {gy:.1f})")

        # ----------------------------
        # 1단계: Constraint A*
        # ----------------------------
        path_to_gate = self.generate_path_astar(
            (sx, sy), (gx, gy),
            use_constraint=True,
            max_cost=0.6
        )

        # ----------------------------
        # 2단계: Standard A*
        #  - 뒤쪽 제약 해제, cost 0.6 유지
        # ----------------------------
        if not path_to_gate:
            print("[Decision] Constraint Failed. Retrying Standard...")
            path_to_gate = self.generate_path_astar(
                (sx, sy), (gx, gy),
                use_constraint=False,
                max_cost=0.6
            )

        # ----------------------------
        # 3단계: Relaxed A*
        #  - 뒤쪽 제약 X, cost 상한 0.85로 완화
        # ----------------------------
        if not path_to_gate:
            print("[Decision] Standard Failed. Retrying Relaxed (Cost < 0.85)...")
            path_to_gate = self.generate_path_astar(
                (sx, sy), (gx, gy),
                use_constraint=False,
                max_cost=0.85
            )

        if not path_to_gate:
            print("[Decision] All A* Failed! Waiting...")
            return

        # ----------------------------
        # 출발 직후 구간 '직선화' 시도
        #  - 시작 지점에서 일정 거리(cut_distance) 떨어진 지점까지
        #    직선으로 갈 수 있으면 경로를 단순화 → 초기 좌우 흔들림 감소
        # ----------------------------
        straightened_path = []
        cut_distance = 3.5
        cut_idx = -1
        for i, p in enumerate(path_to_gate):
            if math.hypot(p[0] - sx, p[1] - sy) > cut_distance:
                cut_idx = i
                break

        can_straighten = False
        if cut_idx != -1:
            cut_point = path_to_gate[cut_idx]
            if self._check_line_of_sight_strict((sx, sy), cut_point):
                can_straighten = True

        if can_straighten:
            cut_point = path_to_gate[cut_idx]
            steps = 10
            for i in range(steps):
                alpha = i / steps
                ix = sx + alpha * (cut_point[0] - sx)
                iy = sy + alpha * (cut_point[1] - sy)
                straightened_path.append((ix, iy))
            straightened_path.extend(path_to_gate[cut_idx:])
            path_to_gate = straightened_path

        # ----------------------------
        # 마지막: 게이트 → 슬롯 중심 직선 보간
        #  - 차량 자세를 슬롯과 평행하게 정렬하는 구간
        # ----------------------------
        path_final = []
        steps = 15
        for i in range(1, steps + 1):
            t = i / steps
            lx = gx + t * (t_cx - gx)
            ly = gy + t * (t_cy - gy)
            path_final.append((lx, ly))

        # 최종 웨이포인트 = A* 경로 + 직선 진입 구간
        self.waypoints = path_to_gate + path_final
        self.planning_done = True

    # Pure-Pursuit 기반 추종 제어
    def get_lookahead_point(self, x, y, yaw, v):
        """
        차량 속도에 따라 lookahead 거리를 동적으로 조절하고,
        그 거리만큼 앞에 있는 waypoint를 반환.
        """
        base_lookahead = 4.0
        lookahead = base_lookahead + 0.8 * v
        lookahead = max(base_lookahead, min(lookahead, 8.0))

        # 현재 위치와 가장 가까운 waypoint 찾기
        min_d = float('inf')
        closest_idx = 0
        for i, (wx, wy) in enumerate(self.waypoints):
            d = math.hypot(wx - x, wy - y)
            if d < min_d:
                min_d = d
                closest_idx = i

        # 그 이후부터 lookahead 이상 떨어진 지점을 목표로 선정
        target_idx = len(self.waypoints) - 1
        for i in range(closest_idx, len(self.waypoints)):
            wx, wy = self.waypoints[i]
            d = math.hypot(wx - x, wy - y)
            if d >= lookahead:
                target_idx = i
                break
        return self.waypoints[target_idx]

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        Pure-Pursuit + 간단한 속도 제어
        - steer : Pure-Pursuit
        - accel/brake : 목표 속도 스케줄링 (슬롯 가까울수록 감속)
        """
        t = float(obs.get("t", 0.0))
        state = obs.get("state", {})
        x, y = float(state.get("x", 0)), float(state.get("y", 0))
        yaw, v = float(state.get("yaw", 0)), float(state.get("v", 0))
        limits = obs.get("limits", {})
        L = float(limits.get("L", 2.8))

        # 이미 주차 완료 상태라면 브레이크 유지
        if self.is_parked:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # 초기 1초 동안은 경로 계획만 하고 정지
        if t < 1.0 or not self.planning_done:
            if not self.planning_done and t > 0.2:
                self.compute_planning(obs)
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        if not self.waypoints:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # Pure-Pursuit 조향
        tx, ty = self.get_lookahead_point(x, y, yaw, v)
        alpha = normalize_angle(math.atan2(ty - y, tx - x) - yaw)
        Lf = math.hypot(tx - x, ty - y)
        if Lf < 0.1:
            Lf = 0.1
        steer = math.atan2(2.0 * L * math.sin(alpha), Lf)

        # 최종 waypoint와의 거리 기반 속도 스케줄링
        last_wp = self.waypoints[-1]
        dist_to_goal = math.hypot(last_wp[0] - x, last_wp[1] - y)

        target_v = 6.0    # 멀리서는 6 m/s로 빠르게 접근
        if dist_to_goal < 8.0:
            target_v = 1.5
        if dist_to_goal < 3.0:
            target_v = 0.8
        if dist_to_goal < 0.3:
            target_v = 0.0

        error = target_v - v
        accel = 0.0
        brake = 0.0

        if v > target_v and v < target_v + 0.5:
            accel = 0.0
            brake = 0.0
        elif v < target_v:
            accel = min(0.5, error * 0.4)
            brake = 0.0
        else:
            if target_v == 0.0:
                brake = 1.0
            else:
                brake = min(0.4, -error * 0.3)

        if dist_to_goal < 0.2:
            self.is_parked = True
            accel = 0.0
            brake = 1.0

        return {"steer": steer, "accel": accel, "brake": brake, "gear": "D"}


planner = PlannerSkeleton()


def handle_map_payload(m):
    planner.set_map(m)


def planner_step(o):
    return planner.compute_control(o)
