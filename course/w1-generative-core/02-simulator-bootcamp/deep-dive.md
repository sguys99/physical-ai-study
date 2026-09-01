# W1-M2 심화 (내부 전개, 전체 표, 운영 세부)

> 이 문서는 [lesson.md](lesson.md)에서 덜어낸 심화 내용입니다. 본문을 먼저 읽으세요.

본문에는 **결과, 직관, 그림, 회사 스택 연결**만 남기고, 계산 단계의 전개와 전체 실측표, 실기 스펙 상세, 클라우드 운영 세부를 여기로 옮겼습니다. 내용은 삭제하지 않았습니다. 읽는 순서만 바뀌었습니다.

| § | 무엇이 있나 | 본문에서 오는 링크 |
|---|---|---|
| §1 | 시뮬레이터 후보별 접촉 모델과 배경 | lesson §2.1 |
| §2 | `mj_step` 다섯 단계의 내부 전개 | lesson §3.2 |
| §3 | 상보성 제약과 soft constraint 완화 | lesson §3.3 |
| §4 | 관절 30개 전체 인덱스 표 | lesson §5.4 |
| §5 | Unitree G1 실기 전체 스펙 | lesson §5.3 |
| §6 | sim2real 갭 씨앗 다섯의 상세 | lesson §5.6 |
| §7 | 클라우드 운영과 예산 최적화 | lesson §6.4 |
| §8 | playground 환경 19개와 보상항 24개 전량 | lesson §7.2 · §7.5 |
| §9 | `unitree_mujoco` 디렉터리와 메시지 목록 | lesson 회사 스택 연결 |
| §10 | API 이름과 렌더링과 설치의 실무 주석 | lesson §3.3 · §6.2 · §6.3 |
| §11 | 유효관성 역산값을 어떻게 읽나 | lesson §5.5 |

---

## 1. 시뮬레이터 후보별 접촉 모델과 배경

본문 §2.1의 비교표에서 접촉 모델과 클라우드 친화도 두 축을 여기로 옮겼습니다.

| 시뮬레이터 | 접촉 모델 | 클라우드 친화 |
|---|---|---|
| **MuJoCo (classic)** | soft-constraint solver, 정확도 우선(`iterations=100`) | ◎ |
| **MJX** | 동일 수식, 처리량을 위해 정확도 하향(`iterations=5`) | ◎ |
| **mujoco_playground** | MJX 상속. 비전 태스크는 MJWarp 배치 렌더러 | ◎ |
| **Genesis** | 강체, 유체, 변형체 다중 물리를 한 엔진에서 다루는 것을 지향 | ○ |
| **Isaac Sim / Isaac Lab** | PhysX | △ (셋업 난도가 높음) |
| PyBullet / Drake | 아래 설명 참조 | ○ |

PyBullet은 진입장벽이 낮아 과거 표준이었으나 신규 로봇 RL 프로젝트는 MuJoCo와 Isaac 쪽으로 이동했습니다. Drake는 접촉 역학과 제약 최적화의 정밀도, 그리고 형식 검증을 지향하는 계열이라 성격이 다릅니다. 둘 다 이 4주에서는 이름과 역할만 알아두면 충분합니다.

Isaac Sim의 렌더링은 RT 코어를 요구합니다. **A100과 V100은 Isaac Sim 렌더링을 지원하지 않습니다.** 클라우드에서 Isaac 계열을 시도하려면 인스턴스 선택 단계에서 이 제약이 먼저 걸립니다.

---

## 2. `mj_step` 다섯 단계의 내부 전개

본문 §3.2가 한 줄씩만 적은 각 단계에서 실제로 어느 필드가 채워지는지입니다.

1. **순기구학**. `qpos`(관절 좌표)로부터 모든 body의 월드 포즈 `xpos[31, 3]`와 `xquat[31, 4]`를 계산합니다. 여기서 렌더링에 필요한 정보도 같이 확정됩니다.
2. **질량행렬과 bias**. $M(q)$를 조립해 `data.M`에 넣고, 코리올리와 원심과 중력을 묶은 bias 항을 `qfrc_bias`에 계산합니다.
3. **제약 탐지**. 충돌 검사로 접촉점을 찾고 관절 한계와 마찰과 등식 제약까지 모아 제약 야코비안 `efc_J`를 만듭니다. 이 단계에서 문제의 크기 `nefc`가 매 스텝 달라집니다.
4. **액추에이터에서 solver로**. `ctrl`을 일반화력 `qfrc_actuator`로 변환한 뒤 제약을 만족시키는 제약력을 반복 최적화로 풉니다(`opt.iterations`회). G1 classic 모델은 Newton solver 100회입니다. MJX 모델도 solver는 같은 Newton이고 반복 횟수만 5회로 줄어듭니다.
5. **적분**. 얻어진 `qacc`로 `qvel`을 갱신하고 `qvel`로 `qpos`를 갱신합니다. 자유 관절의 쿼터니언 부분은 단순 덧셈이 아니라 지수사상으로 갱신됩니다(lesson §3.4).

비용의 대부분이 3번과 4번에 있습니다. 접촉점 개수가 늘면 `nefc`가 커지고 solver가 푸는 문제의 크기가 그대로 커집니다. MJX가 처리량을 위해 `iterations`를 100에서 5로 가장 먼저 깎은 것이 이 구조 때문입니다.

### 2.1 도메인 랜더마이제이션이 `mjModel` 규칙을 깨는 방식

`mjModel`은 원래 모든 환경이 공유하는 상수입니다. 그런데 환경마다 질량과 마찰과 지연을 다르게 주려면 `mjModel`의 일부 필드까지 배치 축으로 올려야 합니다. 강인제어에서 불확실성 집합 $\Delta$를 잡아놓고 그 집합 전체에 안정한 제어기를 설계하는 것과 같은 발상이고, 여기서는 그 집합을 샘플링해서 학습 데이터로 흘려보냅니다. 본격적인 이야기는 W3-M1입니다.

### 2.2 `timestep`과 `iterations`와 `integrator`가 각각 정하는 것

- **`timestep`**은 이산화 주기 $\Delta t$입니다. 작을수록 정확하고 비용은 선형으로 증가합니다
- **`iterations`**는 제약 최적화 solver의 반복 횟수입니다. 부족하면 제약이 덜 만족되어 발이 지면을 파고들거나 미끄러집니다
- **`integrator`**는 적분 스킴이고 G1은 `implicitfast`입니다. 속도 의존 항인 감쇠와 마찰을 암묵적으로 처리해서 뻣뻣한 시스템에서도 큰 `timestep`을 버팁니다. 뻣뻣한 미분방정식에 explicit Euler를 쓰면 $\Delta t$를 아주 잘게 줄여야 하지만 implicit 계열이면 여유가 생기는, 그 이야기 그대로입니다

MJX가 `kp`를 500에서 75로, `kv`를 관절별 값에서 2로 낮춘 것도 같은 맥락입니다. 뻣뻣한 위치 서보는 큰 `timestep`에서 발산하기 쉽고, 이산 시간 서보의 안정 영역이 게인과 시간 간격의 곱에 걸려 있기 때문입니다.

---

## 3. 상보성 제약과 soft constraint 완화

접촉력은 상태의 함수로 미리 쓸 수 없습니다. 세 조건이 동시에 걸립니다.

- 발이 땅을 뚫지 않는다: $\phi(q) \ge 0$
- 접촉력은 밀기만 한다: $f_c \ge 0$
- 떨어져 있으면 힘이 0이다: $\phi \cdot f_c = 0$

세 번째가 상보성(complementarity) 조건입니다. 두 양이 모두 음이 아니고 곱이 0이라는 형태이며, 이걸 매 스텝 풀어서 $f_c$를 얻습니다.

이 하드 제약을 그대로 두면 미분대수방정식(DAE)이 되어 수치적으로 까다롭습니다. MuJoCo는 그것을 **매우 뻣뻣한 상미분방정식**으로 바꿔 풉니다. 접촉을 스프링과 댐퍼로 대체하는 것이고, `solref`가 그 스프링과 댐퍼의 특성(시상수와 감쇠비), `solimp`가 침투 깊이에 따라 임피던스를 어떻게 바꿀지의 프로파일입니다.

대가는 미세한 관통입니다. 발이 지면을 아주 조금 파고듭니다. 이득은 수치 안정성과 미분 가능성입니다. 이 "미분 가능한 접촉"이 MuJoCo가 RL과 궤적 최적화 쪽에서 오래 사랑받은 이유입니다. 기울기가 존재하니까요.

다만 무한 강성을 흉내 내려고 `solref` 시상수를 지나치게 줄이면 시스템이 뻣뻣해져서 `timestep`을 줄여야 하고 그러면 느려집니다. **정확도와 안정성과 속도의 3자 트레이드오프**가 여기서 나옵니다. 뻣뻣한 시스템에 explicit 적분기를 쓰면 시간 간격을 잘게 줄여야 하는 그 상황과 같은 구조입니다.

---

## 4. 관절 30개 전체 인덱스 표

lesson §5.4가 자유 관절과 왼다리와 허리만 실었습니다. 나머지 전부입니다.

**오른다리 6개**

| jnt id | 이름 | type | `qpos` adr | `dof` adr | `ctrl` id | range (rad) |
|---|---|---|---|---|---|---|
| 7 | `right_hip_pitch_joint` | hinge | 13 | 12 | 6 | [-2.531, 2.880] |
| 8 | `right_hip_roll_joint` | hinge | 14 | 13 | 7 | [-2.967, 0.524] |
| 9 | `right_hip_yaw_joint` | hinge | 15 | 14 | 8 | [-2.758, 2.758] |
| 10 | `right_knee_joint` | hinge | 16 | 15 | 9 | [-0.087, 2.880] |
| 11 | `right_ankle_pitch_joint` | hinge | 17 | 16 | 10 | [-0.873, 0.524] |
| 12 | `right_ankle_roll_joint` | hinge | 18 | 17 | 11 | [-0.262, 0.262] |

**왼팔 7개**

| jnt id | 이름 | `qpos` adr | `dof` adr | `ctrl` id | range (rad) |
|---|---|---|---|---|---|
| 16 | `left_shoulder_pitch_joint` | 22 | 21 | 15 | [-3.089, 2.670] |
| 17 | `left_shoulder_roll_joint` | 23 | 22 | 16 | [-1.588, 2.252] |
| 18 | `left_shoulder_yaw_joint` | 24 | 23 | 17 | [-2.618, 2.618] |
| 19 | `left_elbow_joint` | 25 | 24 | 18 | [-1.047, 2.094] |
| 20 | `left_wrist_roll_joint` | 26 | 25 | 19 | [-1.972, 1.972] |
| 21 | `left_wrist_pitch_joint` | 27 | 26 | 20 | [-1.614, 1.614] |
| 22 | `left_wrist_yaw_joint` | 28 | 27 | 21 | [-1.614, 1.614] |

**오른팔 7개**

| jnt id | 이름 | `qpos` adr | `dof` adr | `ctrl` id | range (rad) |
|---|---|---|---|---|---|
| 23 | `right_shoulder_pitch_joint` | 29 | 28 | 22 | [-3.089, 2.670] |
| 24 | `right_shoulder_roll_joint` | 30 | 29 | 23 | [-2.252, 1.588] |
| 25 | `right_shoulder_yaw_joint` | 31 | 30 | 24 | [-2.618, 2.618] |
| 26 | `right_elbow_joint` | 32 | 31 | 25 | [-1.047, 2.094] |
| 27 | `right_wrist_roll_joint` | 33 | 32 | 26 | [-1.972, 1.972] |
| 28 | `right_wrist_pitch_joint` | 34 | 33 | 27 | [-1.614, 1.614] |
| 29 | `right_wrist_yaw_joint` | 35 | 34 | 28 | [-1.614, 1.614] |

좌우 대칭 관절의 range가 부호만 뒤집힌 것에 주의하세요. `left_hip_roll`이 [-0.524, 2.967]이고 `right_hip_roll`이 [-2.967, 0.524]입니다. 좌우 궤적을 복사할 때 부호를 뒤집어야 합니다.

`left_shoulder_roll`과 `right_shoulder_roll`도 같은 관계이고, `yaw` 계열처럼 범위가 대칭인 관절은 부호를 뒤집어도 같은 구간이 나옵니다.

---

## 5. Unitree G1 실기 전체 스펙

lesson §5.3이 DoF와 토크와 질량만 실었습니다. 2026-08-01 확인 기준 나머지입니다.

- 크기는 기립 시 1320 × 450 × 200 mm, 접힘 690 × 450 × 300 mm입니다
- 무릎 최대 토크는 90 N·m(기본)이고 EDU는 120 N·m입니다
- 팔 가반하중은 약 2 kg(기본), 약 3 kg(EDU)입니다
- 배터리는 13S 리튬 9000 mAh로 가동 약 2시간입니다
- 센서는 뎁스 카메라와 3D LiDAR와 4-마이크 어레이입니다
- 온보드 컴퓨트는 기본 8-core CPU이고 고성능 컴퓨트 모듈이 옵션입니다
- 통신은 WiFi 6과 Bluetooth 5.2입니다
- 기본형 가격은 US $13.5K입니다

menagerie 모델의 총질량 33.341 kg은 단순화 모델의 값이고 실기는 배터리 포함 약 35 kg입니다. 질량 분포도 단순화돼 있으므로 이 차이를 그대로 관성 오차로 읽으면 안 됩니다.

---

## 6. sim2real 갭 씨앗 다섯의 상세

lesson §5.6 표의 각 항목을 W1-M1 「흔한 오해」 3의 5대 요인과 연결한 전개입니다.

**1. `forcerange = [0, 0]`이라 토크 제한이 없다.** 실기 무릎은 90 N·m(기본)과 120 N·m(EDU)에서 포화합니다. 시뮬 정책이 무제한 토크로 균형을 잡아놓고 실기에서 포화 때문에 넘어지는 것이 전형적인 실패 경로입니다. 요인은 액추에이터 동특성입니다.

**2. `armature = 0.01`이 전 관절 동일하다.** 반사 관성은 회전자 관성에 감속비의 제곱을 곱한 값이라 관절마다 다릅니다. 이 값은 수치 안정용 대충값에 가깝습니다. 요인은 액추에이터 동특성입니다.

**3. `frictionloss = 0.3`이 전 관절 동일하다.** 관절 쿨롱 마찰은 감속기 종류와 조립 상태와 온도에 따라 달라집니다. 요인은 액추에이터 동특성입니다.

**4. 발 접촉이 `sphere size=0.005`, `friction=0.6`, `condim=3`, `priority=1`이다.** 발바닥이 평면이 아니라 반지름 5 mm짜리 작은 구 몇 개로 근사돼 있습니다. 실기 고무 발바닥의 접촉 패치와 변형과 다르고 마찰 0.6도 바닥 재질마다 다릅니다. `priority=1`이라 접촉 상대의 마찰 대신 이 값이 채택됩니다. 요인은 접촉 모델입니다.

**5. 센서가 넷뿐이고 총질량이 33.341 kg 대 약 35 kg이다.** 시뮬에서는 `data.qpos`로 참값을 공짜로 읽지만 실기에는 그런 것이 없습니다. 질량 분포도 단순화됐습니다. 요인은 센서 노이즈와 상태 추정입니다.

그리고 **지연은 이 모델 파일에 아예 표현돼 있지 않습니다.** 실기에는 센서 읽기부터 정책 추론과 통신과 모터 반영까지 지연이 쌓이는데 시뮬은 기본적으로 0입니다. 도메인 랜더마이제이션에서 지연을 별도로 주입하는 이유가 그것입니다.

---

## 7. 클라우드 운영과 예산 최적화

lesson §6.4의 네 원칙을 제어 용어로 옮기면 이렇게 됩니다.

| 원칙 | 내용 | 제어공학적 번역 |
|---|---|---|
| 인스턴스는 소모품 | 코드와 데이터와 체크포인트는 퍼시스턴트 볼륨에 | 상태를 플랜트가 아니라 관측 가능한 저장소에 둔다 |
| 재현성 | Docker 이미지 또는 버전 고정 환경 | 실험 조건의 파라미터 고정 |
| headless | `MUJOCO_GL=egl`, 결과는 mp4나 png | 관측 채널을 오프라인화 |
| 원격 관측 | 학습 로그는 실험 추적 도구, 장시간 작업은 tmux | 세션 단절이라는 외란에 대한 대책 |

목적함수는 "4주 안에 마일스톤 넷(FSQ 구현, LeRobot 완주, G1 보행과 sim2sim, 캡스톤)을 완주"이고 제약은 GPU 시간 예산입니다.

lesson §6.4에서 결론만 남기고 옮겨온 판단 기준 셋입니다.

- 중간 저장 없이 임시(스팟) 인스턴스에서 학습을 돌리면 회수되는 순간 처음부터 다시입니다. 체크포인트가 곧 안전장치이고, 없으면 개루프로 기계를 돌리는 것과 같습니다
- 논문을 읽는 네 시간 동안 GPU가 켜져 있는 것이 4주 예산에서 가장 큰 단일 낭비입니다. 읽기와 돌리기를 시간대로 분리하세요
- 상한을 정하지 않은 최적화는 최적화가 아닙니다. 주간 GPU 시간 상한을 먼저 정하고 그 안에서 모듈별로 배분합니다

인스턴스 선택은 마스터플랜 부록 A 기준입니다. W1과 W2의 작업(생성모델 toy, LeRobot PushT)은 T4나 A10G나 L4 한 장이면 충분하고, W3의 playground RL 학습은 A10G나 L4나 L40S급을 권장합니다. **A100과 H100은 이 플랜에서 불필요합니다.** Isaac Lab을 선택 과제로 시도할 때만 RT 코어가 있는 GPU(L4, L40S, A10G 계열)가 필요하고, A100과 V100은 Isaac Sim 렌더링을 지원하지 않습니다.

제공자별 구체 절차(인스턴스 생성부터 볼륨 마운트, 드라이버 확인, 첫 mp4 저장까지)는 [`labs/README.md`](labs/README.md)에 있습니다. lesson과 이 문서는 제공자 중립으로 원칙만 다룹니다.

---

## 8. playground 환경 19개와 보상항 24개

### 8.1 `registry.locomotion.ALL_ENVS` 전체 19개

`G1JoystickFlatTerrain`, `G1JoystickRoughTerrain`, `H1InplaceGaitTracking`, `H1JoystickGaitTracking`, `Go1JoystickFlatTerrain`, `Go1JoystickRoughTerrain`, `Go1Getup`, `Go1Handstand`, `Go1Footstand`, `T1JoystickFlatTerrain`, `T1JoystickRoughTerrain`, `ApolloJoystickFlatTerrain`, `BarkourJoystick`, `BerkeleyHumanoidJoystickFlatTerrain`, `BerkeleyHumanoidJoystickRoughTerrain`, `Op3Joystick`, `SpotFlatTerrainJoystick`, `SpotGetup`, `SpotJoystickGaitTracking`

이 중 G1과 H1 태스크는 앞의 4개입니다. 백엔드는 MJX(JAX)와 MuJoCo Warp 양쪽을 지원하고, 비전 태스크는 MJWarp 배치 렌더러를 씁니다.

### 8.2 `cfg.reward_config.scales` 24개 전량

| 그룹 | 항 (개수) |
|---|---|
| **추종** | `tracking_lin_vel`, `tracking_ang_vel` (2) |
| **자세와 안정** | `orientation`, `base_height`, `lin_vel_z`, `ang_vel_xy`, `pose`, `stand_still`, `alive`, `termination` (8) |
| **접촉과 발** | `feet_air_time`, `feet_clearance`, `feet_height`, `feet_phase`, `feet_slip`, `contact_force`, `collision` (7) |
| **정규화와 비용** | `action_rate`, `dof_acc`, `energy`, `torques`, `dof_pos_limits`, `joint_deviation_hip`, `joint_deviation_knee` (7) |

LQR과 MPC 비용함수와의 대응은 이렇게 갈립니다.

- $Q$의 역할은 `tracking_lin_vel`, `tracking_ang_vel`, `orientation`, `base_height`가 맡습니다
- $R$의 역할은 `torques`, `energy`, `action_rate`, `dof_acc`가 맡습니다
- 제약의 소프트 페널티는 `dof_pos_limits`, `collision`, `contact_force`입니다

다른 점은 셋입니다. 첫째로 부호가 뒤집혀 최대화 문제가 됩니다. 둘째로 항이 24개나 되고 서로 스케일이 크게 달라 가중치 튜닝이 사실상 별개의 작업이 됩니다. 셋째가 결정적인데, `feet_air_time`이나 `feet_phase`처럼 접촉 이벤트에 걸린 항은 미분 불가능하거나 이산적입니다. 비용함수가 매끄러우면 MPC로 풀면 되지 그레이디언트 추정에 수백만 샘플을 쓸 이유가 없습니다. 목적함수가 접촉이라는 이산 이벤트를 품고 있는 것이 RL이 필요한 진짜 이유입니다.

가중치가 0인 항이 여럿 섞여 있습니다. 튜닝의 흔적이고, 실습에서 실제 값을 출력해보면 어느 항이 죽어 있는지 바로 보입니다.

LQR과 MPC 비용함수와의 대응은 lesson §7.5에 있습니다. 여기서 덧붙일 것은 스케일 문제입니다. 24항의 크기가 서로 크게 달라 가중치를 정하는 일이 사실상 별개의 최적화가 되고, 그래서 legged RL 논문들이 보상 가중치 표를 부록에 통째로 싣습니다. W3-M1에서 이 표를 직접 만지게 됩니다.

---

## 9. `unitree_mujoco` 디렉터리와 메시지 목록

lesson 「회사 스택 연결」에서 존재만 알아둔 저장소의 세부입니다.

- 지원 로봇은 `a2` `b2` `b2w` `g1` `go2` `go2w` `h1` `h1_2` `h2` `r1` 열 종이고 G1이 여기 들어 있습니다
- 기본 브랜치는 `main`이고 README 파일명이 소문자 `readme.md`입니다
- 최상위 구조는 `doc/` `example/` `simulate/` `simulate_python/` `terrain_tool/` `unitree_robots/`입니다
- `simulate/`는 C++ 구현이고 권장 경로입니다. `simulate_python/`은 파이썬 버전으로 `config.py`, `unitree_mujoco.py`, `unitree_sdk2py_bridge.py`, `test/`로 구성됩니다
- 지원 메시지는 `LowCmd`, `LowState`, `SportModeState`, `IMUState`이고 G1은 `rt/secondary_imu` 토픽을 씁니다
- 현재 버전은 low-level 개발만 지원합니다. 컨트롤러의 sim2real 검증 용도입니다

---

---

## 10. API 이름과 렌더링과 설치의 실무 주석

### 10.1 질량행렬 필드 이름 (lesson §3.3)

MuJoCo 3.11 기준으로 질량행렬 필드는 **`data.M`** 입니다. 예전 문서와 블로그와 구버전 코드에는 `data.qM`으로 나오지만 현재 바인딩에는 그 이름이 없어서 `AttributeError`가 납니다. 희소 저장이라 `d.M.shape == (341,)`이고, 밀집 행렬이 필요하면 `mujoco.mj_fullM(m, d, dst)`로 뽑습니다. 이 시그니처도 구버전의 `mj_fullM(m, dst, qM)`에서 바뀌었습니다. 검색으로 찾은 예제가 안 돌아가면 가장 먼저 의심할 지점입니다.

### 10.2 종료 시 나오는 EGL traceback (lesson §6.2)

`MUJOCO_GL=egl`에서 인터프리터가 종료될 때 `Renderer.__del__`이 `OpenGL.raw.EGL._errors.EGLError`를 `Exception ignored in:` 접두사로 출력하는 경우가 있습니다. **렌더 결과에는 영향이 없습니다.** 그 접두사는 파이썬이 이미 "이 예외는 무시했다"고 알려주는 것이고 프레임은 정상 저장돼 있습니다. 처음 보면 렌더가 실패한 줄 알고 30분을 태우기 좋은 지점이라 미리 적어둡니다.

`osmesa`는 `libOSMesa`가 없으면 `AttributeError: 'NoneType' object has no attribute 'glGetError'`가 납니다. `sudo apt install libosmesa6-dev`로 해결합니다. `glfw`는 디스플레이가 있는 환경에서만 동작합니다(WSLg의 `DISPLAY=:0` 등).

### 10.3 클라우드 셋업 절차

제공자별 구체 절차(인스턴스 생성부터 볼륨 마운트, 드라이버 확인, 첫 mp4 저장까지)는 [`labs/README.md`](labs/README.md)에 있습니다. lesson과 이 문서는 제공자 중립으로 원칙만 다룹니다.

---

## 11. 유효관성 역산값을 어떻게 읽나

lesson §5.5의 $M_{\text{eff}}$ 역산표에 대한 보충입니다.

고관절은 팔꿈치보다 유효관성이 20배 이상 큽니다. 몸 전체를 흔드는 관절과 아래팔만 흔드는 관절의 차이니 당연합니다. 모든 관절에 같은 `kv`를 주면 어떤 관절은 **과감쇠**, 어떤 관절은 **부족감쇠**가 되고, `dampratio` 파라미터는 그걸 각 관절이 알아서 맞추게 하는 장치입니다.

역산값에는 `armature`(lesson §5.6의 갭 요인 2)도 포함돼 있습니다. 정확한 링크 관성만 뽑으려면 `mj_fullM`으로 $M(q)$를 직접 꺼내 보세요. 그 값은 자세에 따라 변하지만 `kv`는 컴파일 시점에 한 번 정해진 상수라는 점도 같이 확인할 만합니다.

역산 절차를 한 번 더 적어둡니다. eq. 3 $k_v = 2\zeta\sqrt{k_p M_{\text{eff}}}$를 $M_{\text{eff}}$에 대해 풀면 $M_{\text{eff}} = (k_v / 2\zeta)^2 / k_p$이고, 임계감쇠라 $\zeta = 1$입니다. 팔꿈치는 $(9.42912002 \div 2)^2 \div 500 = 22.227 \div 500 \approx 0.044$, 어깨는 $(16.72533692 \div 2)^2 \div 500 = 69.934 \div 500 \approx 0.140$, 무릎은 $(15.84701068 \div 2)^2 \div 500 \approx 0.126$, 고관절은 $(43.01068276 \div 2)^2 \div 500 \approx 0.925$입니다.

lesson §5.5의 `<position kp="500" dampratio="1" inheritrange="1"/>` 한 줄은 컴파일되면 `gainprm[0] = kp`, `biasprm = [0, -kp, -kv]`가 됩니다. 액추에이터가 내는 힘 $f = k_p(\texttt{ctrl} - q) - k_v \dot q$의 두 이득이 이 두 배열에서 나오고, `practice/02_g1_inspect.py`가 $k_p$를 `m.actuator_gainprm[aid, 0]`에서, $k_v$를 `-m.actuator_biasprm[aid, 2]`에서 읽는 근거도 이것입니다. `dampratio`는 파일에 그대로 남지 않고 컴파일 시점에 $k_v$로 환산돼 사라집니다.

`inheritrange="1"` 덕분에 `ctrlrange`가 관절 `range`와 자동으로 같아집니다. lesson §5.4 표의 range를 그대로 명령 한계로 쓰면 됩니다.

---

**본문으로 돌아가기** → [lesson.md](lesson.md)
