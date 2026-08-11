# Fixed reader prompts — 1273 set (one shared question, uid `test_0001`)

**Question:** A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

**Options:** A. Inhibition of thymidine synthesis; B. Inhibition of proteasome; C. Hyperstabilization of microtubules; D. Generation of free radicals; E. Cross-linking of DNA  
**Gold:** E


==========================================================================================
## vanilla (no KG)   (frozen/1273/vanilla.json)
==========================================================================================

Solve the multiple-choice medical question. Provide your final answer as a single letter in <a></a> tags.

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

<a>?</a>

==========================================================================================
## cot (no KG)   (frozen/1273/cot.json)
==========================================================================================

Solve the multiple-choice medical question. Think step by step before giving your final answer. State your final answer as a single letter in <a></a> tags. Example: <a>C</a>

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

==========================================================================================
## raw_1hop (unranked 1-hop dump, 10 shown)   (frozen/1273/raw_1hop.json)
==========================================================================================

You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: 2 days
- Top candidate concepts retrieved from SNOMED via raw graph expansion from seeds (NO filtering, NO ranking — listed in graph-traversal order, may contain noise).

  1. Bladder
  2. Family history of transitional cell carcinoma of bladder
  3. Malignant neoplastic process
  4. Malignant neoplasm of bladder
  5. Primary malignant neoplasm of bladder (disorder)
  6. Carcinoma of bladder
  7. Metastatic transitional cell carcinoma of bladder
  8. Spindle cell urothelial carcinoma of urinary bladder
  9. Plasmacytoid urothelial carcinoma of urinary bladder (disorder)
  10. Primary transitional cell carcinoma of anterior wall of urinary bladder

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Rank the candidates and select the most likely answer.

After your reasoning (under 200 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.

==========================================================================================
## raw_2hop (unranked 2-hop dump, 10 shown)   (frozen/1273/raw_2hop.json)
==========================================================================================

You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: 2 days
- Top candidate concepts retrieved from SNOMED via raw graph expansion from seeds (NO filtering, NO ranking — listed in graph-traversal order, may contain noise).

  1. Bladder
  2. Family history of transitional cell carcinoma of bladder
  3. Malignant neoplastic process
  4. Malignant neoplasm of bladder
  5. Primary malignant neoplasm of bladder (disorder)
  6. Carcinoma of bladder
  7. Metastatic transitional cell carcinoma of bladder
  8. Spindle cell urothelial carcinoma of urinary bladder
  9. Plasmacytoid urothelial carcinoma of urinary bladder (disorder)
  10. Primary transitional cell carcinoma of anterior wall of urinary bladder

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Rank the candidates and select the most likely answer.

After your reasoning (under 200 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.

==========================================================================================
## medrag (textbook chunks)   (frozen/1273/medrag.json)
==========================================================================================

You are an expert medical diagnostician. Solve the multiple-choice medical question using the supplied medical-textbook excerpts as supporting evidence.

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

Retrieved textbook excerpts (top-32 BM25 matches from the MedQA source corpus — Harrison, First Aid, Pathoma, etc.):
- Patient symptom duration: 2 days

[Excerpt 1] (InternalMed_Harrison)
h the diagnosis. The following three brief scenarios of a patient with hemoptysis demonstrate three distinct patterns: A 46-year-old man presents to his internist with a chief complaint of hemoptysis. An otherwise healthy nonsmoker, he is recovering from an apparent viral bronchitis. This presentation pattern suggests that the small amount of blood-streaked sputum is due to acute bronchitis, so that a chest x-ray provides sufficient reassurance that a more serious disorder is absent. In the seco

[Excerpt 2] (InternalMed_Harrison)
han pure tone hearing. Multiple sclerosis may present with acute unilateral or bilateral hearing loss; typically, pure tone testing remains relatively stable while speech understanding 220 fluctuates. Isolated labyrinthine infarction can present with acute hearing loss and vertigo due to a cerebrovascular accident involving the posterior circulation, usually the anterior inferior cerebellar artery; it may also be the heralding sign of impending catastrophic basilar artery infarction (Chap. 446).

[Excerpt 3] (InternalMed_Harrison)
gic Assessment,” below), the bone-conduction stimulus is perceived as louder than the air-conduction stimulus. For the Weber test, the stem of a vibrating tuning fork is placed on Unknown Tectorial membrane protein Gel attachment to nonsensory cell Morphogenesis and cohesion Cytoskeletal protein Reversible S-glutathionylation of CHAPTER 43 Disorders of Hearing the head in the midline and the patient is asked whether the tone is heard in both ears or better in one ear than in the other. With a un

[Excerpt 4] (InternalMed_Harrison)
alleus Lobe Middle ear Eustachian tube FIguRE 43-1 Ear anatomy. A. Drawing of modified coronal section through external ear and temporal bone, with structures of the middle and inner ear demonstrated. B. High-resolution view of inner ear. On examination, there is often a perforation of the tympanic membrane filled with cheesy white squamous debris. The

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Cross-reference the retrieved textbook excerpts to support or rule out candidates.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 250 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.


==========================================================================================
## tog (Think-on-Graph beam)   (frozen/1273/tog.json)
==========================================================================================

You are an expert medical diagnostician. Answer the multiple-choice question.
Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?
Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA
Knowledge-graph evidence (entity - relation - entity), retrieved by Think-on-Graph:
Sensorineural hearing loss -[possibly_equivalent_to]- High tone sensorineural hearing loss
Sensorineural hearing loss -[possibly_equivalent_to]- High frequency deafness
Sensorineural hearing loss -[possibly_equivalent_to]- Low tone sensorineural hearing loss
Reason briefly (<150 words), then give the final answer as a single letter in <a></a> tags.

==========================================================================================
## hykge (HO + reasoning chains)   (frozen/1273/hykge.json)
==========================================================================================

### Task Description:
You are a medical expert. Based on relevant medical [Background Knowledge] and your medical knowledge,
answer the [User Query] while adhering to [Answer Requirements].
### Answer Requirements:
1) Take time to think slowly, understand step by step, and answer questions.
2) Clearly state key information in the answer and provide direct and specific answers to user questions.
### Background Knowledge
The retrieved knowledge chains are:
- Cisplatin poisoning →has_causative_agent→ Drug product →causative_agent_of→ Ototoxicity - deafness (& [drug]) →inverse_isa→ Ototoxicity
- Cisplatin poisoning →inverse_isa→ Antineoplastic poisoning →has_causative_agent→ Other antineoplastic agents (product) →has_direct_substance→ Neoadjuvant antineoplastic chemotherapy
- Cisplatin poisoning →has_causative_agent→ cis-Platinum
- PTA - Pure tone audiometry →direct_procedure_site_of→ Auditory structure (body structure) →finding_site_of→ Ototoxicity - deafness (& [drug]) →inverse_isa→ Ototoxicity
- Neoadjuvant antineoplastic chemotherapy ←direct_substance_of← Other antineoplastic agents (product) antineoplastic agents (product) →inverse_isa→ cis-Platinum
- Neoadjuvant antineoplastic chemotherapy →has_direct_substance→ Other antineoplastic agents (product) →inverse_isa→ cis-Platinum
- Sensorineural hearing loss (& [deafness: [high frequency] or [low frequency]) (disorder) →possibly_equivalent_to→ Perceptive hearing loss NOS (disorder) →is_interpreted_by→ Functional observable →is_interpreted_by→ Subjective tinnitus
- Sensorineural hearing loss (& [deafness: [high frequency] or [low frequency]) (disorder) →possibly_equivalent_to→ High tone sensorineural hearing loss →inverse_isa→ [X]Diseases of inner ear (disorder) →inverse_isa→ Ototoxicity
### User Query
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?
Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA
Reason briefly (<150 words), then give the final answer as a single letter in <a></a> tags.

==========================================================================================
## walker = OUR METHOD (cos + 0.3·bc − 0.08·hop, bc=Bhattacharyya)   (frozen/1273/walker.json)
==========================================================================================

You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: 2 days / 7 days
- Top candidate concepts retrieved from SNOMED via graph expansion from seeds (DDx hypotheses + case findings from the vignette only; option text was NOT used as seeds).

  1. [disease] Ringing in ear (finding) (score=0.91: cos=0.78+bc=0.41)
  2. [disease] Low tone sensorineural hearing loss (score=0.87: cos=0.75+bc=0.39)
  3. [disease] Sudden sensorineural hearing loss (score=0.85: cos=0.74+bc=0.89)
  4. [finding] Unspecified sudden hearing loss (disorder) (score=0.81: cos=0.74+bc=0.77)
  5. [finding] Roaring in ear (score=0.75: cos=0.60+bc=0.77)
  6. [disease] Sensorineural hearing loss (& [deafness: [high frequency] or (score=0.75: cos=0.75+bc=0.00)
  7. [disease] Tonal tinnitus (finding) (score=0.74: cos=0.71+bc=0.09)
  8. [disease] Noise effects on inner ear NOS (score=0.71: cos=0.65+bc=0.72)
  9. [finding] Nervous tinnitus (score=0.71: cos=0.77+bc=0.07)
  10. [disease] Perceptive hearing loss NOS (disorder) (score=0.68: cos=0.76+bc=0.00)

How to read each entry:
  `N. [role] name (score=S: cos=C+bc=B)`
  - `score = cos + 0.3·bc − 0.08·hop`: walker's combined ranking signal.
  - `cos`: semantic similarity between the case symptoms and this concept (0..1).
  - `bc` (dur-compat): Bhattacharyya overlap between patient duration and the disease's typical clinical course (0..1; 0 if non-disease).
  - `role`: UMLS TUI-derived category — [disease] / [finding] / [organism] / [procedure] / [anatomy] etc.
  - Use as a cross-reference; do NOT let it override clinical judgment.

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. For each candidate, evaluate compatibility with the patient's symptom duration and other clinical features. High cos but low bc may indicate semantic match without duration fit.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 300 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.


==========================================================================================
## walker_interval (ablation: bc=interval sample, else identical to walker)   (frozen/1273/walker_interval.json)
==========================================================================================

You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

Options:
  A. Inhibition of thymidine synthesis
  B. Inhibition of proteasome
  C. Hyperstabilization of microtubules
  D. Generation of free radicals
  E. Cross-linking of DNA

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: 2 days / 7 days
- Top candidate concepts retrieved from SNOMED via graph expansion from seeds (DDx hypotheses + case findings from the vignette only; option text was NOT used as seeds).

  1. [disease] Low tone sensorineural hearing loss (score=0.91: cos=0.75+bc=0.53)
  2. [disease] Ringing in ear (finding) (score=0.89: cos=0.78+bc=0.36)
  3. [disease] Sensorineural hearing loss (& [deafness: [high frequency] or (score=0.78: cos=0.75+bc=0.12)
  4. [disease] Sudden sensorineural hearing loss (score=0.77: cos=0.74+bc=0.64)
  5. [disease] Tonal tinnitus (finding) (score=0.76: cos=0.71+bc=0.15)
  6. [finding] Nervous tinnitus (score=0.76: cos=0.77+bc=0.23)
  7. [finding] Subjective tinnitus (score=0.75: cos=0.69+bc=0.19)
  8. [finding] Unspecified sudden hearing loss (disorder) (score=0.74: cos=0.74+bc=0.52)
  9. [disease] Recurrent labyrinthitis (disorder) (score=0.73: cos=0.60+bc=0.98)
  10. [disease] Noise effects on inner ear NOS (score=0.72: cos=0.65+bc=0.76)

How to read each entry:
  `N. [role] name (score=S: cos=C+bc=B)`
  - `score = cos + 0.3·bc − 0.08·hop`: walker's combined ranking signal.
  - `cos`: semantic similarity between the case symptoms and this concept (0..1).
  - `bc` (dur-compat): Bhattacharyya overlap between patient duration and the disease's typical clinical course (0..1; 0 if non-disease).
  - `role`: UMLS TUI-derived category — [disease] / [finding] / [organism] / [procedure] / [anatomy] etc.
  - Use as a cross-reference; do NOT let it override clinical judgment.

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. For each candidate, evaluate compatibility with the patient's symptom duration and other clinical features. High cos but low bc may indicate semantic match without duration fit.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 300 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.
