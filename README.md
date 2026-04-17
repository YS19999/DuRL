# Multi-domain generalization few-shot intent recognition with dual representation learning
## Abstract:
Intent recognition stands as one of the key challenges in achieving high-quality human-computer interaction, with few-shot intent recognition specifically addressing data scarcity. Existing approaches assume either identical data domains or single-domain generalization, overlooking the inherent diversity of intent domains in real-world scenarios. Intent expressions across domains often exhibit significant differences, severely impairing a model’s recognition capability in new domains. However, existing methods focus solely on in-domain adaptation or cross-domain transfer, resulting in an isolation between in-domain and cross-domain representation learning. To address this, in this paper, we propose a novel Dual Representation Learning (DuRL) method that simultaneously enhances in-domain and cross-domain representation learning. DuRL improves accuracy for multi-domain generalization few-shot intent recognition by increasing the discriminative power of in-domain samples and reducing cross-domain representation differences. In the in-domain, DuRL utilizes the generalized knowledge from support instances to enhance the representations of query instances. Simultaneously, to reduce confusion between different class representations, we introduce class discrimination regularization to drive representations of distinct classes apart in feature space. In the cross-domain, DuRL employs representations from various domains to perform adversarial learning, thereby obtaining domain-invariant representations. During the adversarial process, the representations from different domains are pulled closer together to confuse the discriminator. In few-shot recognition, DuRL predicts the target class by computing the matching similarity between support and query instances. Theoretical proofs demonstrate the feasibility of enhancing in-domain generalization, while extensive experiments confirm that DuRL achieves state-of-the-art performance. Results show DuRL outperforms the SOTA by 4.39% and 2.15% on average in Clinic150 → Hwu64 of 10-way 1-shot and 3-shot scenarios, respectively.
## Framework
![image](https://github.com/YS19999/DuRL/blob/main/Figure/model.pdf)
## How to train
python main.py
## Set dataset for domain generalization
**cb**: Clinic150→Banking77<br>
**ca**: Clinic150→ACID<br>
**ch**: Clinic150→Hwu64<br>
**bc**: Banking77→Clinic150<br>
**ba**: Banking77→ACID<br>
**bh**: Banking77→Hwu64<br>
