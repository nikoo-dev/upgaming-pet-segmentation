# პროექტის ახსნა და ინტერვიუსთვის მომზადება

ეს არის სასწავლო გზამკვლევი და არა Upgaming-ის სავარაუდო კითხვების ოფიციალური სია. ჯერ თავად გაუშვი პროექტი, წაიკითხე კოდი და შეამოწმე რეალური შედეგები. AI-ის მიერ დაწერილი ტექსტის დაზეპირება კოდის გაგებას ვერ შეცვლის.

## პირველი მოკლე მეცადინეობა: ერთი ფოტო და ერთი mask

ჯერ გახსენი `artifacts/data/audit_examples.png`. მარცხნივ ფოტოა, შუაში annotation, მარჯვნივ annotation overlay. ვარდისფერი არის pet, შავი background, ყვითელი ignored boundary. ეს არის მონაცემის შემოწმება და არა მოდელის prediction.

Raw label **3** ნიშნავს გაურკვეველ კონტურს. მას ვაქცევთ **255**-ად. `cross_entropy(..., ignore_index=255)` ამ პიქსელებს სწავლისას გამოტოვებს; `metrics.confusion` მათ შეფასებისას გამოტოვებს. ამიტომ მაღალი **pet IoU** არ ამტკიცებს ბეწვისა და კიდეების ზუსტ გამოყოფას. 255 მესამე სასწავლო კლასი არ არის: მოდელს მხოლოდ ორი output channel აქვს.

პირველი ნაბიჯი: შენი სიტყვებით ახსენი, რატომ არის ორი output channel, როცა target-ში 0, 1 და 255 გვხვდება. შემდეგ დამოუკიდებლად გაუშვი `python -m pytest tests/test_data_metrics.py -q` და აჩვენე `map_trimap`-ში სად ხდება mapping. ჯერ mapping-ის შეცვლა არ გჭირდება. შემდეგ ეტაპზე დაამატე precision/recall ქვემოთ მოცემული სავარჯიშოს მიხედვით. ახსნა და ცვლილება შესრულებულად ჩაითვლება მხოლოდ მას შემდეგ, რაც თავად გააკეთებ და შედეგს განვიხილავთ.

სამი პიქსელის პატარა მაგალითი შეგიძლია Python-ში გაუშვა. აქ model-ის გარეშე ვქმნით ორ score-ს თითოეული პიქსელისთვის:

```python
import numpy as np
import torch
from petseg.transforms import map_trimap
from petseg.metrics import confusion, scores

target = map_trimap(np.array([[1, 2, 3]], dtype=np.uint8))
logits = torch.tensor([[[[0., 5., 0.]], [[5., 0., 5.]]]])
prediction = logits.argmax(1)[0].numpy()  # [[1, 0, 1]]
print(target)                          # [[1, 0, 255]]
print(confusion(prediction, target))   # [[1, 0], [0, 1]]
print(scores(confusion(prediction, target))["foreground_iou"])  # 1.0
prediction[0, 2] = 0
print(scores(confusion(prediction, target))["foreground_iou"])  # ისევ 1.0
```

პირველ channel-ში background-ის score-ებია `[0, 5, 0]`, მეორეში pet-ის `[5, 0, 5]`. `argmax(1)` ირჩევს channel-ს უფრო დიდი score-ით. მესამე პიქსელის პასუხის შეცვლამ IoU არ შეცვალა, რადგან მის target-ში 255 წერია. ამ მაგალითში IoU=1.0 მხოლოდ ორი შეფასებული პიქსელის სისწორეს აღწერს.

## 1. რას აკეთებს პროექტი?

შემოდის ცხოველის ფოტო. მოდელი თითოეულ პიქსელს ანიჭებს კლასს: **pet** ან **background**. მიღებული ნიღბით ვაჩვენებთ overlay-ს და ვცვლით ფონს.

- **Classification:** ერთი კლასი მთელი სურათისთვის, მაგალითად „კატა“.
- **Object detection:** ობიექტის კლასი და bounding box.
- **Semantic segmentation:** კლასი თითოეული პიქსელისთვის; ყველა ცხოველი ერთი foreground კლასია.
- **Instance segmentation:** ცალკეული ცხოველების ერთმანეთისგან გამიჯვნა. ჩვენი მოდელი ამას არ აკეთებს.

Background replacement აქ არის binary compositing და არა **alpha matting**. ბეწვის ნახევრადგამჭვირვალე კიდეებისთვის ცალკე მიდგომა დაგვჭირდებოდა.

## 2. არქიტექტურა და transfer learning

**MobileNetV3-Large backbone** ფოტოდან გამოყოფს მახასიათებლებს. **LR-ASPP segmentation head** ამ მახასიათებლებით ქმნის ორ score-ს თითოეულ პიქსელზე. ეს score-ები არის **logits** და არა უკვე დამტკიცებული ალბათობები.

Backbone იწყებს ImageNet-ზე ნასწავლი წონებით; binary head ახლიდან ინიციალიზდება. ამიტომ პროექტში ნამდვილად საჭიროა training. `model.py`-ში `build_model(pretrained=True)` გამოიყენება ახალი training run-ისთვის; inference-ზე ორივე weights პარამეტრი None არის და შემდეგ ჩვენი checkpoint იტვირთება.

Logits-ის ფორმაა `[batch, 2, height, width]`. Evaluation-ზე logits ჯერ ვაბრუნებთ ორიგინალ ზომაზე bilinear interpolation-ით, მერე `argmax(dim=1)` ირჩევს კლასს. უკვე მიღებული integer ნიღბის bilinear resize კლასების აზრს დაარღვევდა.

## 3. მონაცემების გზა

წაიკითხე `data.py` და `transforms.py`.

Oxford trimap-ის საწყისი ნიშნებია: **1 = pet**, **2 = background**, **3 = ambiguous boundary**. ჩვენ ვქმნით **1, 0, 255** მნიშვნელობებს. 255 არის `ignore_index` და არც loss-ში მონაწილეობს, არც metric-ში. ეს აუცილებლად ახსენი: ჩვენი კარგი IoU ბეწვის კიდეების მაღალ ხარისხს არ ადასტურებს.

**ერთი რიცხვი, ორი განსხვავებული ფორმატი:** training/evaluation target-ში **255 = ignored boundary**, ხოლო ჩამოსატვირთ `mask.png`-ში **255 = predicted pet** და **0 = predicted background**. Export-ისას შიდა 0/1 prediction მრავლდება 255-ზე, რათა თეთრი foreground გამოჩნდეს. Prediction PNG-ში ignored კლასი საერთოდ არ არის. ამიტომ export PNG პირდაპირ training target-ად არ უნდა გამოიყენო: მის თეთრ pet პიქსელებს loss შეცდომით ignored-ად ჩათვლის.

ფოტო resize ხდება bilinear მეთოდით და ნორმალიზდება ImageNet mean/std-ით. ნიღაბი resize ხდება მხოლოდ nearest-neighbor მეთოდით. ნიღბზე RGB normalization არ გამოიყენება.

Horizontal flip ერთდროულად უნდა შეეხოს ფოტოსა და ნიღაბს. Color jitter მხოლოდ ფოტოს ცვლის. წინააღმდეგ შემთხვევაში target და input ერთმანეთს აღარ ემთხვევა.

ფოტოს EXIF orientation და annotation geometry ერთად მუშავდება. ახალი ატვირთული ფოტო orientation-ის მიხედვით სწორდება და RGB-ში გადადის.

## 4. რატომ train / validation / test?

**Train:** gradient-ებით წონების სწავლა. **Validation:** checkpoint-ის, experiment-ისა და input resolution-ის არჩევა. **Test:** ერთხელ, უკვე დაფიქსირებული არჩევანის საბოლოო შეფასება.

თუ test შედეგის ნახვის შემდეგ პარამეტრებს ვცვლით, test უკვე გადაწყვეტილებებში მონაწილეობს და შეფასება ზედმეტად ოპტიმისტური ხდება. ამას data leakage-ის ერთ-ერთი ფორმა ჰქვია.

Official trainval იყოფა დაახლოებით 80/20 პროპორციით breed-ის მიხედვით. იდენტური RGB შიგთავსის მქონე სურათები ერთ ჯგუფში რჩება. Test-ს დამთხვევები trainval-დან იშლება და აღირიცხება. Near-duplicate-ების სრულად აღმოჩენას არ ვამტკიცებთ. გამოიყენება seed 42 და შენახული split manifest.

## 5. Training და frozen backbone

`CrossEntropyLoss(ignore_index=255)` სწორი კლასის score-ს აძლიერებს. `backward()` ითვლის gradient-ებს, `optimizer.step()` ცვლის trainable წონებს. AdamW-ს head-ისთვის უფრო მაღალი learning rate აქვს, pretrained backbone-ისთვის — უფრო დაბალი.

Frozen backbone-ისთვის მხოლოდ `requires_grad=False` საკმარისი არ არის: **BatchNorm running statistics** `train()` რეჟიმში მაინც იცვლება. ამიტომ A ექსპერიმენტში backbone რჩება `eval()` რეჟიმში, head კი სწავლობს. Offline test ამოწმებს backbone-ის წონებსაც და buffers-საც.

თუ ყველა target პიქსელი ignored არის, საშუალო loss განსაზღვრული არ არის; ასეთ batch-ს ვტოვებთ და ვაღრიცხავთ. Non-finite loss-ზე training უნდა გაჩერდეს.

## 6. მეტრიკები

Confusion matrix-ში რიგია ground truth, სვეტი — prediction: `[[TN, FP], [FN, TP]]`.

- **Pet IoU = TP / (TP + FP + FN)**.
- **Pet Dice = 2TP / (2TP + FP + FN)**.
- **Pixel accuracy = (TP + TN) / ყველა valid პიქსელი**.

თუ ფონია ძალიან ბევრი, სულ background-ის პროგნოზიც მაღალ accuracy-ს მიიღებს. ამიტომ მთავარი metric არის pet IoU. Headline metric ითვლება მთელ dataset-ზე დაგროვილი confusion matrix-ით და არა ფოტოების IoU-ების უბრალო საშუალოთი.

ნულოვანი denominator ნიშნავს undefined შედეგს, JSON-ში null-ს. იგი ავტომატურად 1 არ ხდება. Ignored boundary პიქსელები საერთოდ არ შედის matrix-ში.

რეალური, უკვე frozen C/256 მოდელის ოფიციალური test შედეგია **pet IoU 0.939749**, **mean IoU 0.954040**, **Dice 0.968939**, 3,669 ფოტოზე. შეფასდა 575,035,253 valid პიქსელი; 79,778,469 boundary პიქსელი, ანუ მთლიანის 12.18%, გამოტოვებულია. ამ რიცხვებს ნუ აურევ ორ training ფოტოზე შესრულებულ overfit diagnostic-ში.

Error figure-ის შემოწმებამ annotation-ის პრობლემებიც გამოაჩინა: `Egyptian_Mau_20`-ში ხილული კატა მთლიანად background-ად არის მონიშნული, ხოლო `Persian_273`-ში კატისა და ფონის ნიშნები ვიზუალურად შებრუნებულია. ამიტომ ყველა IoU=0 ავტომატურად model failure არ არის. Official score უცვლელად დავტოვეთ; test annotation-ები არ გავასწორეთ და ახალი „გაწმენდილი“ score არ დაგვითვლია. სწორი file format და label values semantic correctness-ს ვერ ამტკიცებს.

## 7. ექსპერიმენტების დაცვა

A: frozen backbone. B: full fine-tuning. C: B + augmentation. ერთი და იგივე split, seed, input size და მაქსიმალური budget გვაძლევს უფრო გასაგებ შედარებას. ყველა დამოუკიდებლად იწყება; C არ აგრძელებს B-ის ნასწავლ წონებს.

წაიკითხე რეალური `history.csv`, `selection.json` და `results.md`. ახსენი: რომელმა გაიმარჯვა, რამდენ epoch-ზე, როგორ იცვლებოდა train loss და validation IoU. შედეგის მიმართულება წინასწარ არ გამოიგონო. ერთი seed-ით სტატისტიკურ სანდოობას ვერ დავამტკიცებთ.

**Overfitting:** train უმჯობესდება, validation კი აღარ უმჯობესდება ან უარესდება. **Early stopping:** სამი არაგაუმჯობესებული epoch-ის შემდეგ შეჩერება, მინიმუმ სამი სრული epoch-ის პირობით. არჩეულია საუკეთესო validation checkpoint და არა უბრალოდ ბოლო epoch.

**რეალურად შესრულებული შედარება:** სამივე ექსპერიმენტი local CPU-ზე დასრულდა ერთნაირი reduced configuration-ით: 1,000 training image, batch size 2, 12 epoch და 6,000 optimizer step. Validation-ის 736 და ოფიციალური test-ის 3,669 ფოტო არ შემცირებულა. ეს არ არის full-training-data შედეგი ან Colab-ზე შესრულებული run.

| Experiment | საუკეთესო validation pet IoU |
|---|---:|
| A: frozen backbone | 0.802834 |
| B: full fine-tuning | 0.887705 |
| C: fine-tuning + augmentation | 0.888393 |

Validation-only წესით აირჩა **C**. B-მ A-ს დაახლოებით 0.0849 IoU-ით აჯობა. C-სა და B-ს სხვაობა მხოლოდ **0.000688 IoU** არის; ერთი seed-ით ეს augmentation-ის სანდო უპირატესობას ვერ ამტკიცებს. შეგიძლია თქვა, რომ ამ კონკრეტულ run-ში C ოდნავ წინ იყო და ამიტომ წინასწარ არჩეული წესით გაიმარჯვა. არ თქვა, რომ augmentation ყოველთვის უკეთეს შედეგს იძლევა. ყველა run-ის საწყისი model tensor-ების hash ერთნაირი იყო; C არ განაგრძობდა B-ის training-ს.

კონკრეტული checkpoint მაგალითი: A-ს საუკეთესო validation შედეგი მე-10 epoch-ზე ჰქონდა, თუმცა training მე-12 epoch-მდე გაგრძელდა. B-სა და C-ს საუკეთესო epoch მე-12 იყო. ეს გვაჩვენებს, რატომ ვინახავთ ცალკე `best.pt`-ს და `latest.pt`-ს. Checkpoint-ში epoch ინდექსი ნულიდან იწყება, ამიტომ მე-12 epoch-ის მნიშვნელობა 11-ია.

## 8. ოპტიმიზაცია და გაზომვა

იგივე წონებით ვადარებთ 256/192/128 input-ს. პატარა input შეიძლება აჩქარდეს, მაგრამ ხარისხი შემცირდეს. წონების ფაილი არ მცირდება.

არჩეულია ყველაზე პატარა resolution, რომელიც validation pet IoU-ს მაქსიმუმ 0.02 აბსოლუტური ერთეულით ამცირებს და median end-to-end CPU latency-ს აუმჯობესებს. თუ ასეთი არ არის, ვტოვებთ 256-ს.

**Model-only latency** ზომავს forward-ს. **Pipeline latency** დამატებით მოიცავს preprocess-ს, logits resize-ს და mask generation-ს. Disk I/O, PNG encoding და UI rendering გაზომვაში არ შედის. ყველა candidate ერთი CPU/thread count-ით, FP32-ით, warm-up-ის შემდეგ იზომება. p95 აღწერს შედარებით ნელ გამოძახებებს; median — ტიპურ გამოძახებას.

რეალური local CPU benchmark, ერთი thread, FP32, batch 1:

| Input size | Validation pet IoU | Forward median | Pipeline median / p95 |
|---|---:|---:|---:|
| 256 | 0.888393 | 42.92 ms | 105.42 / 135.90 ms |
| 192 | 0.864532 | 28.14 ms | 92.84 / 119.56 ms |
| 128 | 0.461857 | 18.95 ms | 83.45 / 111.72 ms |

192 უფრო სწრაფია, მაგრამ ხარისხის დანაკარგი **0.023861 IoU** გამოვიდა და წინასწარ დადგენილ 0.02 ზღვარს გასცდა. 128-ის დანაკარგი კიდევ უფრო დიდია. ამიტომ დარჩა **256**. ეს არის შესრულებული, მაგრამ მიღების კრიტერიუმით **წარუმატებელი optimization attempt**; შედეგი არ დაგვიმალავს და ზღვარი შედეგების ნახვის შემდეგ არ შეგვიცვლია. Forward-ის დაჩქარება მთელ pipeline-ს იმავე პროპორციით არ აჩქარებს, რადგან original-size logits resize და სხვა ეტაპებიც დროს მოითხოვს. ამ გაზომვიდან კონკრეტული bottleneck-ის ზუსტ წილს profiling-ის გარეშე ვერ დავამტკიცებთ.

შენი სიტყვებით ახსენი: რატომ არ ავირჩიეთ უფრო სწრაფი 192? რატომ არის C-ს 0.000688 უპირატესობა სუსტი დასკვნის საფუძველი? პასუხის ცოდნა ამ დოკუმენტის არსებობით არ დასტურდება — თავად უნდა ახსნა და saved artifacts აჩვენო.

## 9. Checkpoint და გამეორებადობა

Checkpoint შეიცავს model/optimizer state-ს, epoch-ს, step-ს, early-stopping მდგომარეობას, RNG state-ებს, config-ს, split hash-ს და გარემოს ვერსიებს. `latest.pt` განაგრძობს პროცესს, `best.pt` ინახავს საუკეთესო validation მოდელს. SHA-256 ამოწმებს ფაილის იდენტობას და არა მის ML ხარისხს.

Interrupted epoch თავიდან სრულდება. მოწყობილობებსა და ვერსიებს შორის bitwise იდენტურობას არ ვპირდებით. CPU offline test ამოწმებს კონკრეტულ გარემოში uninterrupted და resumed training-ის თანხვედრას.

## 10. Upload-იდან შედეგამდე

`app.py` ამოწმებს checkpoint-ს და decision-ს, კეშავს მოდელს, იღებს JPEG/PNG-ს, ამოწმებს ზომას, ასწორებს EXIF-ს, ქმნის RGB-ს და იძახებს საერთო inference ფუნქციას. იგი ამზადებს tensor-ს, ითვლის logits-ს, აბრუნებს ორიგინალ ზომაზე და ქმნის 0/1 mask-ს. Overlay და composite ცალკე მზადდება. Download mask არის 0/255 PNG.

სცადე ახალი pet ფოტო და ლიცენზირებულ/საკუთარ no-pet ფოტოზე შეცდომის მაგალითი. No-pet მაგალითი აღწერე მხოლოდ qualitative შემოწმებად; ჩვენი dataset არ ამტკიცებს pet-absence rejection-ს.

უკვე შესრულებული ორი qualitative შემოწმება იხილე `artifacts/qualitative/outputs/`-ში. `chelsea`-ს სახე შენარჩუნდა, მაგრამ თხელი whisker-ები ნაწილობრივ დაიკარგა. `astronaut`-ში pet არ ჩანს, თუმცა model-მა 1,051 პიქსელი foreground-ად დატოვა (mask-ის 0.401%). ეს არის **mask occupancy** და არა confidence, IoU ან no-pet benchmark. ფოტოებს reference segmentation mask არ ჰქონდათ. წყაროები და უფლებები წერია `docs/QUALITATIVE_SOURCES.md`-ში.

ნამდვილი model failure-ის ასახსნელად გახსენი `artifacts/reports/error_followup_2.png` და `error_followup_3.png`: `British_Shorthair_8`-ში კატა თითქმის მთლიანად გამოტოვებულია; `Bengal_24`-ში თეთრი cushion foreground-ში მოხვდა; `miniature_pinscher_5`-ში პატარა ძაღლის მხოლოდ ფრაგმენტები დარჩა. ჯერ თავად შეადარე original, target და prediction. ამ აღწერების წაკითხვა არ ნიშნავს, რომ პირადად უკვე შეამოწმე ან შეგიძლია მათი დამოუკიდებლად ახსნა.

## სავარჯიშოები ახსნით

სავარჯიშოები ცალკე project copy-ში შეასრულე. შენახული submission checkpoint, decision და test შედეგები უცვლელად დატოვე. ქვემოთ მოცემული ნაბიჯები რეკომენდებული პრაქტიკაა; შენ მიერ შესრულებულად ჯერ არ ითვლება.

1. **შეცვალე ფონის ფერი.** ჯერ აპში აირჩიე ორი ფერი და ორივეჯერ ჩამოტვირთე mask. შეამოწმე, რომ mask-ების SHA-256 ერთნაირია. შემდეგ ცალკე ასლში შეცვალე `app.py`-ში color picker-ის საწყისი ფერი, თავიდან გაუშვი აპი და ახსენი, რატომ შეიცვალა composite, მაგრამ mask არა. HEX ფერი გადაიქცევა RGB tuple-ად და გადაეცემა `render_outputs`; ახალი training საჭირო არ არის.
2. **დაამატე precision/recall.** `metrics.scores`-ში დაამატე `divide(tp, tp + fp)` და `divide(tp, tp + fn)`. Undefined შემთხვევა ისევ None უნდა იყოს. Hand-calculated test დაამატე, მაგალითად TP=1, FP=1, FN=1 -> ორივე 0.5.
3. **შეცვალე inference size.** საცდელად გაუშვი `petseg predict photo.jpg --size 192`. ახსენი, რატომ არ შეცვლილა model parameters და რატომ უნდა შეადარო quality/speed validation-ზე production default-ის შეცვლამდე.
4. **გაასწორე არასწორი mask mapping.** დროებით შეცვალე 1-ისა და 2-ის mapping მხოლოდ სავარჯიშო ასლში. გაუშვი mapping test და ნახე audit overlay. ახსენი, რატომ შეიძლება loss მაინც იკლებდეს, მიუხედავად იმისა, რომ სასურველი foreground არ ისწავლება. აღადგინე სწორი mapping და ხელახლა გაუშვი tests.

## დამოუკიდებლად შესამოწმებელი კითხვები

შეგიძლია კოდის გარეშე დახატო data flow? შეგიძლია აჩვენო სად იქმნება optimizer და სად ინახება checkpoint? როგორ გაიგებ, რომ ნიღაბი არასწორად არის resize-ებული? რატომ არის backbone.eval() აუცილებელი A-ში? რა მოხდა რეალურად შენს ყველაზე ცუდ test მაგალითში? რომელი AI დახმარება გამოიყენე და პირადად რა გადაამოწმე?

ამ კითხვებზე პასუხები შენს რეალურ შედეგებს უნდა ეყრდნობოდეს. ინტერვიუმდე დამოუკიდებლად შეასრულე მინიმუმ ორი ცვლილება და გაიგე მათი გავლენა.
