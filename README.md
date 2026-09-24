# Tối ưu đội hình và quá trình tái cấu hình cho đàn UAV

Project nghiên cứu bài toán bố trí và tái cấu hình đội hình cho một nhóm UAV.

Với một tập các điểm cần quan sát trên bản đồ, hệ thống phải tìm vị trí thích hợp cho từng UAV sao cho quan sát được nhiều mục tiêu nhất có thể. Việc bố trí đồng thời phải tránh để các UAV đứng quá gần nhau, hạn chế việc nhiều UAV cùng tập trung quan sát lặp lại một khu vực khi không cần thiết, và bảo đảm toàn bộ đội vẫn duy trì được liên lạc.

Sau khi giải được bài toán bố trí một đội hình tĩnh, project tiếp tục mở rộng sang tình huống thực tế hơn: các UAV đang ở một đội hình ban đầu và cần di chuyển sang một đội hình mới. Khi đó không chỉ đội hình cuối cùng phải tốt, mà toàn bộ quá trình di chuyển cũng phải an toàn, không làm mất liên lạc, không gây va chạm và không đi xuyên qua vùng cấm.

Quá trình phát triển được chia thành ba checkpoint:

```text
Checkpoint 01
Mô hình bài toán, cách đánh giá và các thuật toán cơ bản
        ↓
Checkpoint 02
Tối ưu đội hình có xét trực tiếp cấu trúc liên lạc
        ↓
Checkpoint 03
Tối ưu quá trình chuyển đội hình và ghép với bài toán tìm đội hình đích
```

---

# Checkpoint 01: Mô hình bài toán và các thuật toán cơ bản

Checkpoint đầu tiên tập trung vào việc định nghĩa bài toán rõ ràng trước khi xây dựng thuật toán phức tạp hơn.

Một bài toán gồm:

- kích thước bản đồ;
- vị trí các điểm cần quan sát;
- trọng số của từng điểm;
- số lượng UAV;
- bán kính quan sát của UAV;
- bán kính liên lạc giữa các UAV;
- khoảng cách an toàn tối thiểu giữa hai UAV.

Một nghiệm là tập vị trí của toàn bộ UAV:

```text
X = [(x1, y1), (x2, y2), ..., (xN, yN)]
```

Từ đội hình này, hệ thống tính các tiêu chí chính sau.

## Mức độ quan sát

Một mục tiêu được xem là đã được quan sát nếu có ít nhất một UAV nằm trong bán kính quan sát của nó.

Ngoài tỷ lệ mục tiêu được quan sát thông thường, hệ thống còn hỗ trợ trọng số cho từng mục tiêu. Nhờ vậy những mục tiêu quan trọng hơn có thể đóng góp nhiều hơn vào kết quả cuối cùng.

Trong code, đại lượng này được lưu dưới tên:

```text
weighted_coverage_ratio
```

## Mức độ quan sát dư thừa

Nếu nhiều UAV cùng quan sát một mục tiêu thì phần quan sát lặp lại được tính là dư thừa.

Dư thừa không phải lúc nào cũng xấu, nhưng trong bài toán hiện tại nó được xem là một chi phí. Nếu một khu vực đã được quan sát đủ, việc đưa thêm UAV tới đó có thể làm lãng phí khả năng mở rộng sang vùng chưa được quan sát.

## Khả năng duy trì liên lạc

Hai UAV có thể liên lạc trực tiếp nếu khoảng cách giữa chúng không vượt quá bán kính liên lạc.

Từ đó tạo thành một đồ thị liên lạc:

```text
UAV = node
liên lạc trực tiếp = edge
```

Đội hình được xem là còn liên lạc nếu toàn bộ đồ thị vẫn liên thông. Một UAV không nhất thiết phải liên lạc trực tiếp với tất cả UAV khác, nhưng phải tồn tại đường truyền thông qua các UAV trung gian.

## Khoảng cách an toàn

Khoảng cách giữa mọi cặp UAV phải lớn hơn hoặc bằng khoảng cách an toàn tối thiểu.

Nếu hai UAV đứng quá gần nhau thì đội hình bị xem là vi phạm ràng buộc va chạm.

## Hàm đánh giá

Các thành phần trên được gộp thành một điểm đánh giá chung:

```text
fitness
= weighted_coverage
- redundancy_weight × redundancy_excess
- connectivity_weight × connectivity_deficit
- collision_weight × collision_ratio
```

Trong đó:

- mức độ quan sát được tối đa hóa;
- quan sát dư thừa bị trừ điểm;
- mất liên lạc bị phạt;
- vi phạm khoảng cách an toàn bị phạt.

Checkpoint 01 sử dụng ba thuật toán cơ bản.

### Random Search

Sinh nhiều đội hình ngẫu nhiên và giữ lại đội hình có điểm đánh giá tốt nhất.

Thuật toán này đơn giản nhưng hữu ích để làm mốc so sánh.

### Connected Greedy

Đặt UAV lần lượt vào các vị trí có lợi cho việc quan sát, đồng thời yêu cầu UAV mới vẫn có thể kết nối với phần đội hình đã được xây dựng.

### Vanilla Genetic Algorithm

Sử dụng thuật toán di truyền thông thường:

```text
Khởi tạo quần thể
    ↓
Đánh giá
    ↓
Chọn lọc
    ↓
Lai ghép
    ↓
Đột biến
    ↓
Lặp lại
```

Checkpoint 01 chủ yếu trả lời câu hỏi:

> Có thể mô hình hóa bài toán, đánh giá một đội hình và chạy các thuật toán cơ bản trên cùng một hệ thống hay không?

Sau khi phần nền này hoạt động, một hạn chế rõ ràng xuất hiện.

Thuật toán di truyền thông thường chỉ biết mạng liên lạc bị hỏng sau khi một nghiệm đã được tạo ra và bị trừ điểm. Nó không sử dụng trực tiếp cấu trúc liên lạc để hướng dẫn quá trình tìm kiếm.

Đây là lý do Checkpoint 02 được xây dựng.

---

# Checkpoint 02: Tối ưu đội hình có nhận biết cấu trúc liên lạc

Checkpoint 02 đặt câu hỏi:

> Nếu thuật toán tiến hóa hiểu trực tiếp cấu trúc liên lạc giữa các UAV, thay vì chỉ phát hiện và phạt sau khi mạng đã bị chia cắt, chất lượng nghiệm có cải thiện không?

Ngoài các thuật toán của Checkpoint 01, checkpoint này bổ sung thêm:

```text
Particle Swarm Optimization
NSGA-II
GraphAwareGA
```

Trong đó `GraphAwareGA` là thuật toán chính được đề xuất.

Luồng xử lý cơ bản:

```text
Quần thể hiện tại
      ↓
Đánh giá
      ↓
Chọn lọc
      ↓
Lai ghép có xét cấu trúc liên lạc
      ↓
Đột biến có định hướng
      ↓
Sửa lỗi liên lạc
      ↓
Sửa lỗi khoảng cách an toàn
      ↓
Quần thể tiếp theo
```

## Lai ghép có xét cấu trúc liên lạc

Trong thuật toán di truyền thông thường, vị trí UAV có thể được lấy từ hai nghiệm cha gần như độc lập.

Điều này dễ làm mất các UAV đang đóng vai trò cầu nối quan trọng.

`GraphAwareGA` sử dụng thêm thông tin từ đồ thị liên lạc để hạn chế việc lai ghép phá hỏng những cấu trúc kết nối có ích đã tồn tại.

## Bảo vệ UAV cầu nối

Một UAV được xem là điểm khớp nếu việc loại nó khỏi đồ thị có thể làm mạng liên lạc bị chia thành nhiều phần.

Thiết kế ban đầu giảm xác suất hoặc biên độ đột biến đối với những UAV này.

Ý tưởng là tránh việc một thay đổi ngẫu nhiên làm mất cầu nối quan trọng.

Cơ chế này được giữ lại để kiểm tra bằng ablation. Kết quả sau đó không cho thấy đóng góp đủ rõ, nên sang Checkpoint 03 nó không còn được bật mặc định.

## Đột biến hướng tới vùng chưa được quan sát

Một phần đột biến không hoàn toàn ngẫu nhiên.

Nếu còn các mục tiêu chưa được quan sát, thuật toán có thể hướng UAV về các vùng này để tăng khả năng cải thiện nghiệm.

Nhờ vậy quá trình tìm kiếm không chỉ dựa vào nhiễu ngẫu nhiên mà còn sử dụng thông tin trực tiếp từ bài toán.

## Sửa mạng liên lạc

Sau khi lai ghép hoặc đột biến, đội hình có thể bị chia thành nhiều nhóm không còn liên lạc được với nhau.

Thay vì chỉ trừ điểm, thuật toán cố dịch chuyển các nhóm lại gần nhau cho tới khi mạng liên lạc được nối lại.

Việc di chuyển cả một nhóm giúp giữ tương đối cấu trúc bên trong của nhóm đó.

## Sửa khoảng cách an toàn

Nếu hai UAV đứng quá gần nhau, chúng được dịch ra xa nhau.

Do việc sửa khoảng cách có thể ảnh hưởng tới liên lạc và ngược lại, hai quá trình sửa có thể cần được áp dụng luân phiên.

Checkpoint 02 vì vậy chuyển cách xử lý từ:

```text
Sinh nghiệm
    ↓
Vi phạm ràng buộc
    ↓
Trừ điểm
```

sang:

```text
Sinh nghiệm
    ↓
Sử dụng cấu trúc bài toán để hướng dẫn tìm kiếm
    ↓
Sửa các vi phạm
    ↓
Đánh giá
```

Kết quả của checkpoint này là một bộ tối ưu đội hình tĩnh đủ mạnh để tiếp tục sử dụng trong bài toán lớn hơn.

---

# Checkpoint 03: Tái cấu hình đội hình

Hai checkpoint trước mới giải bài toán:

> Đội hình tốt nên nằm ở đâu?

Tuy nhiên trong thực tế các UAV đang ở một đội hình hiện tại.

Chúng không thể xuất hiện ngay tại đội hình mới.

Checkpoint 03 vì vậy mở rộng bài toán thành:

```text
Đội hình hiện tại A
        ↓
Tìm đội hình đích B
        ↓
Di chuyển từ A tới B
```

Trong suốt quá trình di chuyển, hệ thống vẫn phải bảo đảm:

- mỗi UAV không vượt quá tốc độ tối đa;
- toàn bộ đội vẫn giữ được liên lạc;
- các UAV không va chạm với nhau;
- UAV không đi ra ngoài bản đồ;
- UAV không đi xuyên qua vùng cấm.

Checkpoint 03 được chia thành hai lớp.

```text
Lớp 1
Biết trước A và B
→ tìm cách di chuyển từ A tới B

Lớp 2
Biết A
→ tự tìm B tốt
→ đồng thời kiểm tra B có thể đi tới được hay không
```

---

# Gán UAV vào vị trí đích

Các UAV được xem là tương đương nhau.

Vì vậy UAV số 1 không bắt buộc phải đi tới vị trí số 1 trong đội hình B.

Trước khi lập quỹ đạo, hệ thống tìm cách gán UAV hiện tại vào các vị trí đích sao cho quãng đường của UAV phải đi xa nhất là nhỏ nhất.

Bài toán đầu tiên là:

```text
minimize max distance(A_i, B_assignment(i))
```

Sau khi tìm được mức khoảng cách lớn nhất tốt nhất, thuật toán Hungarian được dùng để giảm tổng quãng đường trong các cách gán có cùng mức tối ưu đó.

Việc này giúp giảm thời gian chuyển đội hình ngay từ bước gán vị trí.

---

# Lập kế hoạch chuyển đội hình

Thuật toán chính của phần này là:

```text
BackboneTransitionPlanner
```

Ý tưởng chính là không cần bảo vệ toàn bộ các liên kết trong mạng.

Chỉ cần bảo vệ một tập liên kết đủ để toàn bộ mạng vẫn liên thông.

Tại mỗi bước:

```text
Đội hình hiện tại
      ↓
Tạo đồ thị liên lạc
      ↓
Chọn cây khung làm xương sống
      ↓
Tính hướng di chuyển mong muốn
      ↓
Điều chỉnh chuyển động để giữ xương sống
      ↓
Kiểm tra va chạm và vùng cấm
      ↓
Giảm bước di chuyển nếu cần
      ↓
Đội hình tiếp theo
```

## Cây khung liên lạc

Với mỗi liên kết giữa hai UAV, hệ thống tính phần khoảng cách còn dư trước khi vượt quá bán kính liên lạc:

```text
slack(i, j) = Rc - distance(i, j)
```

Liên kết càng ngắn thì phần dư càng lớn và càng an toàn.

Từ các liên kết hiện tại, thuật toán tạo một cây khung có tổng phần dư lớn.

Cây này chỉ gồm `N - 1` liên kết nhưng vẫn nối được toàn bộ UAV.

Do đó nếu tất cả các liên kết trong cây vẫn được giữ thì toàn bộ đội vẫn liên lạc được.

Cây khung được tính lại sau mỗi bước nên đội hình không bị khóa cứng vào cấu trúc ban đầu.

---

# Kiểm tra liên lạc trong cả đoạn chuyển động

Chỉ kiểm tra liên lạc ở đầu và cuối một bước là chưa đủ.

Hai UAV có thể hợp lệ tại hai đầu nhưng khoảng cách giữa chúng vượt giới hạn ở giữa đoạn.

Planner sử dụng tính chất của khoảng cách Euclid để bảo đảm rằng nếu một liên kết của cây khung hợp lệ ở cả đầu và cuối đoạn chuyển động tuyến tính, thì nó cũng hợp lệ trong toàn bộ khoảng thời gian ở giữa.

Nhờ vậy khả năng liên lạc được bảo vệ liên tục, không chỉ tại các thời điểm rời rạc.

---

# Kiểm tra va chạm liên tục

Va chạm cũng được kiểm tra trên toàn bộ đoạn chuyển động.

Ví dụ:

```text
Ban đầu:    hai UAV cách xa nhau
Cuối bước:  hai UAV vẫn cách xa nhau
Ở giữa:     hai đường bay cắt nhau
```

Nếu chỉ kiểm tra hai đầu thì trường hợp này sẽ bị bỏ sót.

Hệ thống tính khoảng cách nhỏ nhất giữa hai UAV trong suốt đoạn chuyển động và từ chối bước đi nếu khoảng cách này nhỏ hơn mức an toàn.

---

# Tránh vùng cấm

Checkpoint 03 hỗ trợ các vùng cấm hình chữ nhật.

Mỗi vùng cấm có thể được mở rộng thêm một khoảng an toàn trước khi kiểm tra.

Nếu đường thẳng từ UAV tới vị trí đích không đi qua vùng cấm thì UAV tiếp tục di chuyển trực tiếp.

Nếu bị chắn:

```text
Vị trí hiện tại
      +
Vị trí đích
      +
Các góc của vùng cấm
      ↓
Đồ thị nhìn thấy
      ↓
Tìm đường đi ngắn
      ↓
Chọn điểm trung gian đầu tiên
```

Điểm trung gian chỉ được dùng tạm thời.

Ở bước tiếp theo hệ thống tính lại đường đi. Khi vị trí đích có thể đi thẳng tới được, UAV sẽ bỏ điểm trung gian và tiếp tục đi trực tiếp.

Hiện tại vùng cấm chỉ ảnh hưởng tới chuyển động.

Project chưa giả định rằng vật cản làm suy giảm liên lạc hoặc che khuất vùng quan sát.

---

# Từ chuyển đội hình sang bài toán kết hợp

Sau khi phần chuyển từ A tới B hoạt động độc lập, nó được ghép trở lại với bộ tối ưu đội hình của Checkpoint 02.

Cách đơn giản nhất là:

```text
GraphAwareGA
    ↓
Tìm đội hình B
    ↓
BackboneTransitionPlanner
    ↓
Di chuyển từ A tới B
```

Cách này được cài đặt trong:

```text
StaticThenTransition
```

Vấn đề là bước tìm B hoàn toàn không biết B có khó đi tới hay không.

Một đội hình có thể rất tốt khi đứng yên nhưng không phù hợp với trạng thái hiện tại của swarm.

Đây là lý do `TransitionAwareGA` được xây dựng.

---

# TransitionAwareGA

Mỗi cá thể trong thuật toán vẫn chỉ biểu diễn đội hình đích B:

```text
[(x1, y1), ..., (xN, yN)]
```

Quỹ đạo không được nhét trực tiếp vào nhiễm sắc thể.

Mỗi đội hình B được đánh giá theo hai bước:

```text
Đội hình B
    ↓
Đánh giá khả năng quan sát và các ràng buộc tĩnh
    ↓
Nếu không hợp lệ → loại
    ↓
Lập kế hoạch A → B
    ↓
Đánh giá quá trình di chuyển
    ↓
So sánh với các nghiệm khác
```

Nhờ vậy đội hình cuối không chỉ cần tốt mà còn phải thực sự có thể đi tới được từ trạng thái hiện tại.

---

# Ưu tiên chất lượng đội hình trước

Phiên bản hiện tại không dùng một phép cộng trọng số duy nhất để quyết định mọi thứ.

Nếu gộp trực tiếp chất lượng quan sát và thời gian di chuyển vào một điểm, có thể xảy ra trường hợp một đội hình quan sát kém hơn đáng kể vẫn thắng chỉ vì nó nằm gần vị trí hiện tại.

Do đó cách chọn nghiệm mặc định là:

```text
Chuyển đội hình hợp lệ
        ↓
Mức độ quan sát
        ↓
Chất lượng đội hình tĩnh
        ↓
Thời gian chuyển đội hình
        ↓
Tổng quãng đường
```

Mức độ quan sát và chất lượng đội hình được chia thành các khoảng nhỏ theo:

```text
performance_tolerance = 0.005
```

Thời gian và quãng đường chỉ được dùng để phân biệt những nghiệm có chất lượng gần nhau.

Ví dụ, một đội hình quan sát được 95% mục tiêu không nên thua một đội hình chỉ quan sát được 70% chỉ vì đội hình thứ hai gần hơn.

Ngược lại, nếu hai đội hình có chất lượng gần tương đương thì đội hình có thể đạt tới nhanh hơn sẽ được ưu tiên.

Hệ thống vẫn tính:

```text
joint_fitness
```

để phục vụ phân tích và so sánh:

```text
joint_fitness
= static_fitness
- time_weight × normalized_time
- travel_weight × normalized_travel
- infeasible_penalty
```

Nhưng trong chế độ mặc định, giá trị này không trực tiếp quyết định nghiệm cuối.

---

# Điểm tiến gần vùng quan sát

Cách đánh giá phủ theo kiểu có hoặc không tạo ra một vấn đề cho quá trình tìm kiếm.

Ví dụ một UAV có thể cách vùng quan sát của một mục tiêu:

```text
200 m → chưa quan sát được
50 m  → chưa quan sát được
1 m   → vẫn chưa quan sát được
```

Nếu chỉ dùng tỷ lệ quan sát thì cả ba trạng thái đều có điểm giống nhau.

Thuật toán không biết rằng UAV đang đi đúng hướng.

Vì vậy Checkpoint 03 bổ sung một đại lượng phụ:

```text
coverage_potential
```

Đại lượng này tăng dần khi UAV tiến gần tới vùng có thể quan sát được mục tiêu.

Nó chỉ được dùng để hướng dẫn quá trình tìm kiếm.

Kết quả cuối vẫn được đánh giá bằng tỷ lệ mục tiêu thực sự được quan sát.

---

# Di chuyển một nhóm UAV

Một UAV không phải lúc nào cũng có thể tự chạy tới vùng chưa được quan sát.

Nếu nó đang đóng vai trò relay, việc kéo riêng UAV đó đi có thể làm mạng bị đứt. Sau đó bước sửa liên lạc lại kéo nó trở về.

Để xử lý trường hợp này, thuật toán cho phép di chuyển cả một nhóm UAV đang liên kết với nhau.

```text
Cụm mục tiêu chưa được quan sát
        ↓
Chọn UAV làm điểm kéo
        ↓
Lấy các UAV lân cận trong đồ thị liên lạc
        ↓
Kéo cả nhóm về phía mục tiêu
```

UAV gần điểm kéo di chuyển nhiều hơn, các UAV phía ngoài di chuyển ít hơn.

Nhờ vậy cả chuỗi relay có thể dịch chuyển cùng nhau thay vì bắt một UAV tách khỏi mạng.

---

# Hướng dẫn tìm kiếm theo cách xác định

Đột biến ngẫu nhiên vẫn được giữ để tạo đa dạng.

Tuy nhiên việc một cụm mục tiêu lớn có được khám phá hay không không nên phụ thuộc hoàn toàn vào random seed.

Phiên bản hiện tại vì vậy chủ động tìm các cụm mục tiêu chưa được quan sát.

Với mỗi cụm, thuật toán thử nhiều cách:

```text
cụm mục tiêu
× UAV làm điểm kéo
× số bước lân cận trong đồ thị
× độ lớn của bước di chuyển
```

Các nghiệm có triển vọng được đưa trực tiếp vào quần thể của thế hệ tiếp theo.

Luồng mỗi thế hệ hiện tại gần như:

```text
Đánh giá quần thể
      ↓
Giữ lại các nghiệm tốt
      ↓
Sinh thêm nghiệm có định hướng
      ↓
Chọn cha mẹ
      ↓
Lai ghép
      ↓
Đột biến
      ↓
Sửa ràng buộc
      ↓
Đột biến theo nhóm
      ↓
Quần thể tiếp theo
```

Random seed vẫn làm các lần chạy khác nhau, nhưng nó không còn là yếu tố duy nhất quyết định hướng tìm kiếm.

---

# Tinh chỉnh nghiệm cuối

Sau khi thuật toán di truyền kết thúc, hệ thống không lấy ngay cá thể đứng đầu quần thể làm kết quả.

Một số nghiệm tốt và khác nhau được giữ lại làm ứng viên cuối.

Mặc định:

```text
finalists = 4
```

Từng ứng viên tiếp tục được tinh chỉnh bằng các bước di chuyển có chủ đích về vùng còn thiếu quan sát.

Trong quá trình tinh chỉnh, thuật toán có thể tạm thời đi qua một trạng thái chỉ cải thiện tín hiệu tìm kiếm, nhưng nghiệm chính thức chỉ được cập nhật khi chất lượng thực sự tốt hơn.

Sau đó hệ thống tiếp tục loại bỏ những chuyển động không cần thiết nếu việc loại bỏ không làm giảm chất lượng đội hình.

Toàn bộ quá trình này vẫn thuộc một lần chạy duy nhất.

```text
1 seed
    ↓
1 quần thể
    ↓
1 quá trình tiến hóa
    ↓
Tìm kiếm có định hướng
    ↓
Tinh chỉnh
    ↓
Nghiệm cuối
```

Phiên bản hiện tại không chạy nhiều lần rồi chọn kết quả may mắn nhất.

Mục tiêu là làm cho bản thân thuật toán ổn định hơn trước sự thay đổi của seed.

---

# Kết quả của hệ thống

Một kết quả của bài toán tái cấu hình gồm:

```text
Đội hình đích B
+
Quỹ đạo di chuyển từ A tới B
```

Các chỉ số được ghi lại gồm:

- tỷ lệ mục tiêu được quan sát;
- mức quan sát dư thừa;
- tính hợp lệ của đội hình;
- thời gian chuyển đội hình;
- tổng quãng đường di chuyển;
- khả năng hoàn thành quá trình chuyển;
- khả năng duy trì liên lạc;
- khoảng cách nhỏ nhất giữa các UAV trong quá trình di chuyển;
- khả năng tránh vùng cấm;
- hiệu quả thời gian;
- thời gian chạy của thuật toán.

---

# Trực quan hóa

Checkpoint 03 có phần trực quan hóa riêng để quan sát quá trình chuyển đội hình.

Ví dụ:

```powershell
python run_visualize_reconfiguration.py `
    --profile showcase `
    --algorithm aware `
    --budget standard `
    --seed 0
```

Kết quả gồm:

```text
HTML tương tác
PNG tĩnh
```

Giao diện hiển thị:

- đội hình ban đầu;
- đội hình đích;
- vị trí hiện tại của UAV;
- quỹ đạo đã đi;
- mạng liên lạc hiện tại;
- cây liên lạc đang được bảo vệ;
- các điểm cần quan sát;
- các điểm hiện đang được quan sát;
- vùng cấm;
- thời gian đã trôi qua;
- trạng thái liên lạc;
- khoảng cách nhỏ nhất giữa các UAV.

HTML cũng có chế độ chỉnh trực tiếp đội hình B.

Có thể kéo từng UAV để quan sát các chỉ số như mức độ quan sát, liên lạc, khoảng cách an toàn và vùng cấm thay đổi như thế nào.

Chế độ này chủ yếu phục vụ phân tích và debug.

Nó không chạy lại thuật toán lập quỹ đạo ở phía Python sau mỗi lần kéo.

---

# Cài đặt

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Cài thêm các thư viện dùng cho test:

```powershell
python -m pip install -r requirements-dev.txt
```

Chạy test:

```powershell
python -m pytest -q
```

---

# Các lệnh chính

## Benchmark đội hình tĩnh

```powershell
python run_benchmark.py --profile quick --seeds 5
```

## Ablation

```powershell
python run_ablation.py --seeds 10
```

## Benchmark quá trình chuyển A → B

```powershell
python run_transition_benchmark.py --profile quick
```

## Benchmark kết hợp tìm đội hình và chuyển đội hình

```powershell
python run_reconfiguration_benchmark.py --profile quick --seeds 3
```

## Domain / literature baselines

Final CP3 benchmark hiện bao gồm ba UAV-domain adapters gần bài toán hơn
PSO/NSGA-II:

- `jocc_cpgs_2026_then_transition`: centralized projected-gradient JOCC;
- `jocc_dpgs_2026_then_transition`: distributed/local-information JOCC;
- `r2c_ise_aaai26_then_transition`: buffered virtual-force expansion từ R2C.

Các implementation này được ghi rõ là **literature adapters** sang mô hình
weighted target-point 2D của project, không phải bit-for-bit reproduction
khi paper gốc dùng sensing/control model khác. Chi tiết và citation nằm trong
`docs/DOMAIN_BASELINES.md`.

---
## Final CP3 evidence

Final benchmark và ablation dùng 24 paired seeds mặc định. Sensitivity dùng budget
riêng nhỏ hơn vì grid lớn hơn nhiều.

```powershell
python run_cp3_final.py --phase benchmark
python run_cp3_final.py --phase ablation
python run_cp3_final.py --phase sensitivity
python run_cp3_final.py --phase statistics
python run_complexity_scaling.py --seeds 5
```

Benchmark trên target map bên ngoài dùng interchange CSV có cột
`x,y[,weight]`:

```powershell
python run_external_benchmark.py targets.csv --width 1000 --height 1000
```

Repo cũng kèm một snapshot tọa độ địa lý thật từ danh sách điểm trú ẩn FEMA ở
Bay County, Florida. GeoJSON WGS84 được tự động chiếu sang mét; sức chứa sau
thiên tai hoặc sức chứa sơ tán được dùng làm trọng số:

```bash
python run_external_benchmark.py \
    data/external/fema_bay_county_shelters.geojson \
    --name fema_bay_county \
    --seeds 24 \
    --output-dir results/cp3_final/external_fema_bay
```

Mặc định riêng cho GeoJSON địa lý là bán kính cảm biến 3000 m, bán kính liên
lạc 7000 m và lề bản đồ 1000 m. Có thể truyền lại các giá trị này bằng
`--sensing-radius`, `--communication-radius` và `--margin-m`. Lệnh trên chạy
đủ các baseline giống benchmark CP3 chính.

Để chạy một seed của cả năm mô hình và xem mọi đường đi trong **một file HTML**:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
OPENBLAS_NUM_THREADS=1 \
.venv/bin/python run_compare_trajectories.py \
    data/external/fema_bay_county_shelters.geojson \
    --name fema_bay_county \
    --seed 0 \
    --output results/cp3_final/fema_all_models.html
```

Mở `results/cp3_final/fema_all_models.html`, chọn mô hình bằng các thẻ phía
trên biểu đồ rồi dùng nút Play/Pause hoặc thanh thời gian. Script chạy lại đúng
một seed vì `results.csv` của benchmark chỉ lưu chỉ số, không lưu toàn bộ tọa
độ đường đi. Có thể thêm `--subframes 2` để chuyển động mượt hơn, đổi lại file
HTML sẽ lớn hơn đáng kể.

Chỉ cập nhật snapshot khi chủ động tạo một phiên bản benchmark mới:

```bash
python download_fema_shelters.py \
    --state FL \
    --county BAY \
    --output data/external/fema_bay_county_shelters.geojson
```

## Chạy showcase

```powershell
python run_visualize_reconfiguration.py `
    --profile showcase `
    --algorithm aware `
    --budget standard `
    --seed 0 `
    --subframes 10 `
    --frame-ms 60
```

---

# Trạng thái hiện tại

## Desktop simulator thuần Python

Ứng dụng Dear PyGui cho phép sinh/chỉnh bản đồ thành phố 2D, chọn thuật toán,
chạy tối ưu ở tiến trình nền và phát quỹ đạo theo thời gian thực mà không cần
trình duyệt, Node.js hay một máy chủ riêng:

```bash
.venv/bin/python run_simulator_ui.py
```

Để dùng bản Pyglet/OpenGL cho cửa sổ mô phỏng lớn:

```bash
.venv/bin/python run_pyglet_simulator.py
```

Bản Pyglet vẽ mục tiêu, vật cản, UAV, liên kết và vệt bay theo các batch được
giữ lại trên GPU. Giao diện mặc định rộng tối đa 1500×880; dòng trên cùng hiển
thị riêng FPS, thời gian phát lệnh vẽ và thời gian trình bày khung hình. Có thể
dùng chuột cho mọi điều khiển, hoặc nhấn Space để phát/dừng, R để chạy tối ưu,
G để sinh map và Home để tua về đầu.

Simulator mặc định dùng hai bộ đệm để cửa sổ được compositor cập nhật ổn định.
GNOME/X11 trên máy thử nghiệm hiện giới hạn thao tác đổi buffer ở khoảng 20 FPS;
đây là tốc độ hiển thị thật. Tùy chọn `--single-buffer` chỉ dành cho chẩn đoán:
nó có thể báo FPS cao nhưng cửa sổ không được compositor làm mới đầy đủ.

Trong cửa sổ có thể:

- sinh bản đồ dễ/trung bình/khó theo seed;
- thêm mục tiêu, thêm vùng cấm hoặc xóa bằng chuột;
- lưu/mở bản đồ JSON;
- đổi số UAV và các bán kính sensing/liên lạc;
- chạy một trong năm thuật toán benchmark;
- phát, dừng, tua và đổi tốc độ mô phỏng;
- bật/tắt vùng sensing, cạnh liên lạc và vệt bay;
- xem coverage, redundancy, quãng đường, feasibility và lịch sử chạy.

Thuật toán chạy bằng một Python process riêng nên không khóa render loop của UI.
Nếu planner không trả quỹ đạo đã kiểm chứng, simulator giữ UAV tại đội hình A
thay vì vẽ một đường thẳng chưa được chứng nhận an toàn.

## Kiểm chứng đường bay qua obstacle

### Tìm formation và đường bay với phản hồi

```python
from src.proposed import HierarchicalReconfiguration

optimizer = HierarchicalReconfiguration(time_budget_sec=3.0)
solution, runtime = optimizer.solve(problem, seed=0)
# Khi có trạng thái UAV mới, tạo ReconfigurationProblem mới với start_positions
# theo đúng thứ tự ID của UAV, rồi truyền nghiệm trước để tái sử dụng đường:
next_solution, runtime = optimizer.solve(updated_problem, seed=0, warm_start=solution)
```

Thuật toán mới dùng Greedy tránh obstacle để khởi tạo, local search theo nhiều
thứ tự cập nhật để sinh một shortlist formation, rồi lập đường cho các ứng viên.
Ứng viên không tìm được đường trong ngân sách được đánh dấu `not_found_within_budget`,
không kết luận không thể tới. Planner có thể dịch endpoint tối đa 20m trên mỗi
trục, giảm tối đa 1 điểm phần trăm coverage so với đề xuất; chỉ nhận endpoint thực
tế nếu coverage/fitness tốt hơn nghiệm đã giữ và toàn bộ đường được
kiểm chứng. Thời gian bay/travel được báo cáo nhưng không dùng để chọn nghiệm.

Nghiệm tốt nhất luôn gồm cả formation và trajectory hợp lệ. Khi hết ngân sách,
trả nghiệm đã kiểm chứng (có thể là giữ nguyên A nếu chưa tìm được cải thiện).
Đường của lần chạy trước được kiểm tra lại với vị trí, obstacle và target hiện tại
trước khi tái sử dụng. Đây là API hỗ trợ replanning; chưa phải bộ điều khiển MPC
có mô hình động lực học UAV. Planner vẫn dùng tối ưu L-BFGS-B và kiểm định đoạn
liên tục, chưa thay bằng SCP. Ngân sách là cooperative, không bảo đảm hard realtime.

```bash
python run_hierarchical_benchmark.py --profile all --seeds 3 --budget 3
```

Khoảng cách chống va chạm mặc định giữa UAV là **3 m**, áp dụng cả ở formation
đích và xuyên suốt đường bay (kiểm tra liên tục giữa các waypoint). Đây không
phải khoảng cách phân tán đội hình mong muốn. Các profile và script mô phỏng
dùng chung `DEFAULT_MIN_SEPARATION` trong `src/problem.py`; cấu hình tùy chỉnh
vẫn có thể truyền `min_separation` khác. Không thêm penalty phân tán mới hay
thay đổi khoảng cách tránh obstacle/bán kính liên lạc. Bộ tối ưu đường vẫn có
biên mềm 0.5 m để hướng tìm kiếm an toàn; kiểm định cứng dùng đúng 3 m.
Manifest benchmark ghi khoảng cách thực tế của từng case. Kết quả cũ dùng
55–75 m không so sánh trực tiếp được với kết quả cấu hình mới; hãy dùng thư
mục output mới khi chạy lại (kể cả lệnh tạo hard cases từ showcase).

Hierarchical quét controller cục bộ trên các đội hình đề xuất (tối đa 0.08 s/lượt,
tổng tối đa 20% ngân sách), không bắt buộc tiêu 0.9 s cho Greedy trước.
Nếu quét nhanh thất bại hoàn toàn, dành tối đa 25% ngân sách để tạo thêm đội hình
với các thứ tự local-search khác, kiểm tra đường ngay sau mỗi đề xuất mới.
Không tự động ưu tiên thử sâu Greedy trước nữa. Sau đó mỗi lượt tìm đường sâu
được dùng trọn tối đa `attempt_time_sec`, không chia tiếp cho nhiều bộ tìm đường.
Nếu vẫn chưa có đường cải thiện, giữ lại 35% thời gian còn lại cho fallback;
nếu đã có, tái sử dụng đường để nâng coverage.
Khởi tạo cả cụm và nới đích là fallback khi còn thời gian sau các
lượt chính, luôn nằm trong deadline tổng. Mọi đường đều phải qua kiểm định liên tục; đường tâm cụm
không phải chứng nhận an toàn. Chỉ nhận nghiệm tốt hơn nghiệm đã kiểm chứng.
Đây vẫn là L-BFGS-B, chưa triển khai SCP/RRT hay bảo đảm tìm được baseline
trong deadline. Diagnostics ghi `phase`, `method`, `allocated_sec` cho từng lượt.

Có thể thêm `--target-coverage 0.90` vào lệnh benchmark để dừng khi nghiệm đã
kiểm chứng đạt ít nhất 90% coverage. Mặc định tắt: không đánh đổi chất lượng
lấy thời gian một cách ngầm định. Tham số API tương ứng là `target_coverage`.
Diagnostics ghi `stop_reason`, coverage và thời gian của bước phục hồi.

Kết quả nằm ở `outputs/hierarchical/`: CSV cho cùng baseline Greedy và planner,
JSON ghi cấu hình/phản hồi, ảnh và HTML hoạt ảnh của showcase. Greedy deterministic
chạy một lần mỗi case, không áp deadline; các seed của thuật toán mới đổi thứ tự
local search. Mô hình hiện chỉ tính sensing và radio theo khoảng cách: obstacle
chặn đường UAV, chưa mô hình hóa che khuất cảm biến hoặc sóng vô tuyến.

### Incremental: giữ trạng thái qua các chu kỳ

`IncrementalReconfiguration` là API riêng; benchmark hierarchical cũ không đổi.
Đây là incremental local search + waypoint optimization, **không phải GA**.

```python
from dataclasses import replace
from src.proposed import IncrementalReconfiguration

optimizer = IncrementalReconfiguration(step_budget_sec=0.1)
optimizer.initialize(problem, seed=0)
plan, elapsed = optimizer.improve_step()
# Lặp improve_step() để tiếp tục công việc, không gọi initialize() lại.
# Khi có telemetry mới (giữ nguyên thứ tự/identity UAV):
updated = replace(problem, start_positions=measured_positions)
optimizer.update_state(updated)
plan, elapsed = optimizer.improve_step()
safe_plan = optimizer.get_plan()
```

Lưu tiến độ greedy theo UAV, generator local search, hàng đợi mục tiêu, số lượt
đã thử và waypoint đang tối ưu dở. Các waypoint dở chỉ là warm-start, không bao
giờ được trả để thực thi. L-BFGS-B giữ waypoint nhưng không giữ Hessian/history
giữa các lát tính toán. Có giới hạn hàng đợi/restarts/lượt thử; `search_exhausted`
không có nghĩa tối ưu toàn cục. `target_coverage` tùy chọn cho phép dừng sớm.

`update_state()` kiểm tra lại đường còn lại từ vị trí đo, có thể truyền
`path_index` là waypoint đã bay tới trên **plan hiện hành**. Không truyền thì
chọn cấu hình gần nhất rồi kiểm định lại. Nếu đường cũ không hợp lệ, dùng hold
đã kiểm chứng; nếu ngay cả trạng thái đo cũng không hợp lệ, báo lỗi và không
trả đường cũ. Thay target/obstacle/ràng buộc sẽ reset các đánh giá tìm kiếm cũ.
Thay vị trí giữ tiến độ tìm đội hình và các waypoint dùng làm warm-start.

```bash
# Tĩnh: đánh giá chất lượng tích lũy qua 50 chu kỳ
python run_incremental_benchmark.py --rc 150 --seed 3 --cycles 50 --budget-ms 100

# Bay mô phỏng lý tưởng: tiến 1 waypoint mỗi chu kỳ rồi cập nhật/replan
python run_incremental_benchmark.py --rc 150 --seed 3 --cycles 50 --budget-ms 100 --execute-steps 1 --output-dir outputs/incremental_flight
```

JSON ghi thời gian initialize, improve, update và toàn chu kỳ (kể cả kiểm định
độc lập); log tách `current_coverage` (vị trí hiện tại) và `goal_coverage` (đội
hình đích có đường hợp lệ). Trường JSON `coverage` cũ vẫn là coverage đích.
NPZ lưu đoạn đã thực thi và kế hoạch còn lại. Khi cập nhật đúng tại waypoint,
không chèn lại một waypoint chờ trùng; khi lệch khỏi đường vẫn kiểm định đoạn
nối về waypoint. `path_index` đã thực thi được xử lý cả khi đó là một đoạn chờ.
Ngân sách cooperative:
một thao tác tìm kiếm/kiểm định có thể vượt mốc. Chưa có mô hình gia tốc, sai số
bám đường, controller hoặc cam kết prefix khi bay thật; đây là planner hình học,
không phải bộ điều khiển triển khai trực tiếp lên UAV. Không bảo đảm hội tụ tối
ưu sau hữu hạn chu kỳ; không giả định coverage tăng khi môi trường thay đổi.

### Kiểm tra riêng planner với formation cố định

```bash
python run_obstacle_planner_benchmark.py --seeds 3
```

Lệnh này giữ nguyên formation Greedy obstacle-aware trên
`showcase_dense_bridge`, so sánh controller cục bộ với planner mới và chạy
CP3 tích hợp. Kết quả CSV, ảnh và kiểm định trajectory nằm trong
`outputs/global_formation_path/`.

`BackboneTransitionPlanner` thử controller cục bộ trước. Nếu bị kẹt, planner
tối ưu đồng thời các waypoint của toàn swarm, giữ cố định formation A/B.
Mỗi đường trả về phải vượt qua kiểm tra đoạn liên tục về obstacle, khoảng
cách UAV, tốc độ, bounds và spanning tree chứng nhận kết nối. Planner dừng
ngay khi tìm được đường hợp lệ; chưa tối ưu thời gian bay/travel. Đây vẫn là
phương pháp heuristic: tìm không ra đường không chứng minh đường không tồn tại.

CP3 chạy bước này một lần cho formation Greedy tránh obstacle sau vòng GA,
không chạy lại trong từng fitness evaluation. Có thể dùng
`TransitionAwareGA(use_global_planning=False)` hoặc
`BackboneTransitionPlanner(enable_global_planning=False)` để đối chiếu.
Ngân sách planner mặc định là 25 waypoint và tối đa 800 vòng tối ưu; đây
không phải cam kết deadline hard real-time.

Qua ba checkpoint, project đã phát triển từ bài toán:

```text
Các điểm cần quan sát
        ↓
Tìm vị trí UAV
```

thành:

```text
Đội hình hiện tại A
        ↓
Tìm đội hình đích B
        ↓
Đánh giá khả năng quan sát
        ↓
Kiểm tra và sửa các ràng buộc
        ↓
Lập kế hoạch di chuyển A → B
        ↓
Ưu tiên các đội hình có chất lượng tốt
        ↓
Tinh chỉnh nghiệm
        ↓
Đội hình đích + quỹ đạo an toàn
```

Trọng tâm hiện tại vẫn là chất lượng của đội hình cuối và độ ổn định của quá trình tìm kiếm.

Thời gian chuyển đội hình, tổng quãng đường và thời gian chạy của thuật toán đã được đo và đưa vào hệ thống, nhưng chưa được ưu tiên cao hơn chất lượng quan sát.

Các bước tiếp theo có thể tập trung vào:

- giảm thêm độ nhạy theo random seed;
- cải thiện khả năng tìm đội hình trong môi trường có vật cản;
- tối ưu thời gian chuyển đội hình sau khi chất lượng đội hình đã ổn định;
- giảm thời gian chạy của thuật toán;
- mở rộng sang môi trường động hoặc tái cấu hình trực tuyến.
