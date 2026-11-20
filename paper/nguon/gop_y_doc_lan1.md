# Góp ý khi đọc bản v10 (11.2026) — gom lại, sửa một lần sau khi đọc xong

## Chung
- [Deep] Cách viết quá học thuật, câu dài nhồi nhiều ý, phải đọc vài lần mới hiểu (ví dụ 2 câu trong Abstract). Cần viết đơn giản, tự nhiên hơn — có thể phải sửa cả bài.
- [Deep] Nhiều chỗ chưa chắc có nên ghi hay ghi như vậy không; vài chỗ chưa hiểu → giải thích từng đoạn trước, sửa sau.

- [Deep, chỉnh lại] Gộp ý liên quan trong một câu là được; chỉ tách khi từ 3 ý trở lên. Vấn đề chính là diễn đạt mơ hồ, đọc ra nhiều nghĩa (ví dụ "while keeping each site's images and disease mix") → viết sao cho người đọc hiểu đúng một nghĩa bài muốn nói.
- [Deep] Giọng thận trọng, khiêm tốn như các paper lớn: điều phổ biến thì viết "is known to / is often reported to"; điều chưa chắc (cơ chế BN, nguyên nhân phần gap còn lại, DeiT…) viết thành giả thuyết "we hypothesize… based on…", không khẳng định tuyệt đối.

- [Deep] Câu dễ hiểu lầm chủ thể/ý định: "DP works only on the last layer" đọc thành "DP chỉ hoạt động ở lớp cuối", trong khi ý là "ta chỉ áp DP vào lớp cuối, vì…". Rà toàn bài.
- [Deep, nghiêm trọng] Cụm từ tự đặt không có trong hiểu biết chung, dùng như thuật ngữ: "the device effect", và tương tự ("device share", "device part", "device-mixing control", "private design", "FL gap"…). Hoặc thay bằng mô tả thẳng, hoặc định nghĩa rõ một lần ngay khi dùng và giữ thật ít. Rà toàn bài.
- [Deep] Nhận xét của Claude ở Abstract phần lớn là thừa; chỉ giữ: diễn đạt mơ hồ, thuật ngữ tự đặt, khẳng định tuyệt đối.

## Abstract
- [Claude tự thấy] Câu 2 vừa nêu mục tiêu vừa nêu dữ liệu, hai ý nên tách.
- [Claude tự thấy] Câu 3 dùng "mixing imaging devices across sites" mà chưa nói trộn để làm gì; "four controlled comparisons point to the batch-norm statistics that the server averages as the reason" gói 3 khái niệm chưa giải thích (so sánh có kiểm soát, batch-norm, server lấy trung bình).
- [Claude tự thấy] "accuracy gap" ở Abstract khác tên "FL gap" trong thân bài.

## Introduction + Related work (Deep: không quan trọng lắm, sửa nhẹ theo các lỗi chung)
- Khẳng định tuyệt đối: "No single clinic has", "is less accurate", "comes from devices alone", "traces… to BN", "avoids this part", "known to suffer most".
- Mơ hồ/ẩn dụ: "comes at a price on two sides", "argued from the design", "trace the accuracy loss", "hides the advantage", "strong choice".
- Cụm tự đặt: device-mixing control, device effect, key test, device part / the rest, training budget, gap split by cause, build clients.
- LỖI NỘI DUNG: FedBN nghiên cứu feature shift (ảnh khác kiểu giữa các site), không phải lệch nhãn → sửa câu "These studies vary the label mix".

## 3 Method
- LỖI NỘI DUNG: "After 30 rounds the final model is scored" mâu thuẫn với 3.2 "keep the round with the best validation accuracy" (và Fed-ISIC dùng 9/27 vòng, chấm vòng cuối).
- LỖI NỘI DUNG (quan trọng): trộn nguồn không chỉ trộn máy chụp mà trộn cả quần thể bệnh nhân, cách chụp của từng phòng khám (Queensland khác Vienna). Gọi là "devices" là nói quá → nói "image source, mainly the imaging device" và nêu bằng chứng máy chụp (hai nguồn Vienna khác máy) như giả thuyết có cơ sở.
- "The images differ more than the labels": đã bỏ bằng chứng về nhãn (JS divergence) nên câu này không còn được chứng minh → bỏ hoặc thêm lại số.
- Giả thuyết viết như sự thật: "BN is the layer that records the shift, and the averaged statistics fit no site"; "so any difference… comes from devices"; "Noise spread over millions of weights drowns the signal".
- Mơ hồ: "uses the same boxes in one place", "attacked for membership", "whose private form", "the devices behind each site", "differs from the baseline in its layers rather than in scale" (thật ra khác cả kiến trúc lẫn chuẩn hoá — phải nói thẳng), "We apply DP to the last one or two blocks and to the classifier alone" (ý: thử 3 mức), "spends its own privacy budget".
- Cụm tự đặt/nhãn: device-mixing control, FL gap, device share, device effect, private design, private baseline, proposed design (3 nhãn cho phần riêng tư là quá nhiều), federation.
- Thuật ngữ thả vào không giải thích: group normalization, class-weighted cross-entropy, reference models (RMIA), ε (nhỏ = riêng tư hơn).
