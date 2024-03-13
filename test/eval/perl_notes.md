# Solution 분석용 Perl script notes

## HPWL 계산

`perl hpwl.pl /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.nodes /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.pl /home/junetech/data/dp20240312_2/result_rs1001/adaptec1/adaptec1.gp.pl /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.nets`

- Argument 순서: \*.nodes \*.pl \*.gp.pl \*.nets
- HPWL 출력 형태

  ```sh
  Total HPWL: 73171427
  ```

  - 모든 net weight가 1임을 가정하고 단순합
  - scaled HPWL 값이 아님

### 수정 사항

- Line 287: 출력 형태 수정. Logic에 영향 X

## Legality check

`perl legal2.pl /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.nodes /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.pl /home/junetech/data/dp20240312_2/result_rs1001/adaptec1/adaptec1.gp.pl /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.scl`

- Argument 순서: \*.nodes \*.pl \*.gp.pl \*.scl
- 위반 출력 형태

  ```sh
  ERROR SUMMARY.....
  Type    Occurrences
  0       0
  1       0
  2       0
  3       0
  ```

### 수정 사항

- Line 39: 단순 오타 수정. Logic에 영향 X

## Density check

`perl check_density_target.pl /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.nodes /home/junetech/data/dp20240312_2/result_rs1001/adaptec1/adaptec1.gp.pl /home/junetech/data/ispd_benches/ispd2005/adaptec1/adaptec1.scl 1`

- Argument 순서: \*.nodes \*.gp.pl \*.scl target_density
- 결과 출력 형태

  ```sh
  Total 8010 (90 x 89) bins. Target density: 1.000000
  Violation num: 0 (0.000000)     Avg overflow: 0.000000  Max overflow: 0.000000
  Overflow per bin: 0.000000      Total overflow amount: 0.000000
  Scaled Overflow per bin: 0.000000
  ```

### 수정 사항

- Line 37: argument로 받은 density target이 '1 이상'이면 0.5의 값을 사용하도록 강제되어있는데, 이 기준을 '1 초과'로 변경
- Line 614: violation이 하나도 없으면 0으로 나누기 에러가 발생하는데, if 구문으로 nviolation이 0이 아닐때만 나누기하도록 변경
- Line 584: 한 개의 bin이라도 그 면적에 비해 node 총 면적이 1을 초과하는 경우 error를 발생하게 하는데, 필요에 따라 comment-out

## Density check (for newblue1)

- *일부 내용*을 제외하면 위의 일반 density check 코드와 동일
- 필요 없는 처리로 판단됨. 안 쓸 예정
- *일부 내용*: Large movable node를 fixed node로 취급하게 속성 바꾸기
  - transformed.list라는 파일을 읽어서 값을 사용하게 되어있음
    - transformed.list에는 node ID가 나열
    - 해당 node ID는 problem instance data 중 terminal(fixed) node 중에 섞여있는 large movable node의 ID
