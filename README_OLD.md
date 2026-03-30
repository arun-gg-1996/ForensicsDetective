# Forensics Detective

## Project Information
- **Created By:** Justin Del Vecchio
- **Version:** 0.2
- **Date Created:** September 14, 2025
- **Last Updated:** September 19, 2025

## Target Community of Interest
Cybersecurity forensics investigators. This research would benefit professionals who need to analyze digital artifacts and trace their origins, including:

- **Insider threat investigators** - tracking employee access patterns and identifying suspicious activities within organizations
- **Digital forensics analysts** - determining the provenance and authenticity of digital documents like PDFs, images, and other files
- **Incident response teams** - reconstructing attack timelines and understanding how threats moved through systems
- **Corporate security teams** - investigating data breaches and understanding what information was accessed or exfiltrated

## Research Goal
This project aims to develop a toolbox of machine learning-backed forensics investigation tools for cybersecurity professionals. The first tool in development is a PDF Provenance Detector that analyzes PDF documents to determine their creation method - whether they were exported from applications like Microsoft Word or Google Docs, or programmatically generated using libraries such as Python's PDF creation tools. By identifying the generation source, cybersecurity teams can gain valuable context during investigations, potentially revealing information about threat actors' toolsets, document authenticity, and attack methodologies. This capability could also aid in insider threat identification, such as detecting employees with multiple company document PDFs exported from unauthorized systems like Google Docs when corporate policy requires approved platforms.

## Summary

This research addresses a novel application of machine learning to digital forensics through the development of a PDF Provenance Detector. The core methodology involves converting PDF documents to binary image representations and using image classification techniques to identify the creation software used to generate the PDFs.

The experimental approach centers on creating a controlled dataset of 400 documents with identical content but different PDF generation pathways. Wikipedia articles serve as the content source, ensuring diverse yet standardized text across multiple domains including science, history, geography, and technology. These articles are converted to Word documents, then exported to PDF using Microsoft Word's native PDF generation engine. The same content is subsequently uploaded to Google Docs and exported as PDFs using Google's PDF generation system, creating two distinct classes for binary classification.

The key innovation lies in converting the binary structure of PDFs into grayscale images, where each byte value (0-255) corresponds to a pixel intensity. This visualization approach transforms the abstract binary patterns inherent in different PDF generation engines into visually distinguishable signatures. Different PDF creators embed unique metadata structures, apply distinct compression algorithms, and utilize varying font rendering techniques that manifest as detectable patterns in the binary-to-image representation.

This methodology addresses a significant gap in current digital forensics capabilities. While existing expert systems rely heavily on metadata analysis for document provenance, such information can be easily manipulated or stripped by sophisticated actors. The binary visualization approach operates at a deeper structural level, detecting generation signatures that are inherently embedded in the PDF creation process and resistant to simple metadata manipulation.

The research objectives include: (1) demonstrating the feasibility of binary-to-image conversion for PDF classification, (2) achieving statistically significant classification accuracy between Word-generated and Google Docs-generated PDFs, (3) evaluating the robustness of the approach against metadata stripping, and (4) establishing a framework for expanding the toolbox to additional document creation applications and file types.

## Supporting Data Sets

### Primary Dataset: Wikipedia-Derived Document Corpus

**Source Documents (wikipedia_docs/)**
- **Count**: 398 Microsoft Word documents (.docx format)
- **Content Source**: Wikipedia articles spanning diverse academic domains
- **Topic Categories**: Science & Technology, History, Geography, Arts & Literature, Philosophy, Sports, and General Knowledge
- **Content Standardization**: Identical textual content across all documents to ensure controlled experimental conditions
- **Filename Convention**: Underscored naming (e.g., `Python_programming_language.docx`) for command-line compatibility

**Word-Generated PDFs (word_pdfs/)**  
- **Count**: 398 PDF documents
- **Generation Method**: Microsoft Word's native "Export as PDF" functionality
- **PDF Engine**: Microsoft Word's internal PDF generation library
- **Metadata Signature**: Contains Word-specific creator tool signatures and structural patterns
- **File Size Range**: Varies based on content length (typical range: 50KB - 500KB)

**Google Docs-Generated PDFs (google_docs_pdfs/)**
- **Status**: Pending generation
- **Planned Count**: 398 PDF documents (matching source document count)
- **Generation Method**: Google Docs "Download as PDF" functionality  
- **PDF Engine**: Google's cloud-based PDF generation service
- **Control Methodology**: Identical content to Word documents, uploaded via Google Drive API

### Dataset Characteristics

**Content Diversity**: Articles cover topics from `3D_printing` to `Yosemite_National_Park`, ensuring varied vocabulary, formatting complexity, and document structures that test classifier robustness across different content types.

**Experimental Controls**: 
- Identical source content eliminates content-based classification bias
- Consistent formatting across document pairs
- Standardized export settings for each PDF generation method
- Preserved document structure (headings, paragraphs, basic formatting)

**Binary Classification Setup**: The dataset enables direct comparison between two PDF generation engines using identical content, creating a controlled environment for detecting binary-level differences in PDF creation signatures.

### Utility Scripts

**convert.py**: Automated conversion utilities for batch processing documents through different PDF generation pathways, ensuring consistent methodology across the experimental dataset.

## References

### Literature Review and Related Work

This research builds upon established techniques in digital forensics while addressing a significant gap in PDF provenance detection. The literature review reveals three key areas of related work that inform this project's methodology and novelty.

### Binary-to-Image Classification Techniques

The foundational technique of converting binary files to images for classification has been extensively validated in malware analysis. Recent studies demonstrate the effectiveness of transforming executable malware files into grayscale images for classification without requiring file execution, achieving accuracy rates of 98.96% for static analysis and 97.32% for dynamic analysis (Computational Intelligence and Neuroscience, 2022). The methodology involves converting binary content to grayscale PNG images using hexadecimal representation, then applying convolutional neural networks for pattern recognition. Advanced implementations using ResNet50, Xception, and EfficientNet-B4 achieve classification accuracies up to 92.48% while reducing computation time by 95% compared to traditional dynamic analysis methods.

**Key References:**
- [Digital Forensics for Malware Classification: Binary Code to Pixel Vector Transition](https://pmc.ncbi.nlm.nih.gov/articles/PMC9050294/) - PMC Database
- [Malware Analysis Using Visualization of Binary Files](https://www.researchgate.net/publication/262350024_Malware_analysis_method_using_visualization_of_binary_files) - ResearchGate
- [Byte Visualization Method for Malware Classification](https://dl.acm.org/doi/abs/10.1145/3409073.3409093) - ACM Digital Library

### PDF Forensics and Document Authentication

Current PDF forensics research focuses primarily on malware detection and metadata analysis. Recent 2024 studies introduce advanced techniques including PDFObj2Vec representation learning and Graph Isomorphism Networks for structural feature extraction. The field recognizes PDF as the dominant medium for malware distribution (66.6% of malicious email attachments), driving development of sophisticated detection systems using BERT, Word2Vec, and graph neural networks. However, these approaches target malicious content detection rather than creator identification through binary signatures.

**Key References:**
- [Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis](https://arxiv.org/html/2506.17162) - arXiv
- [PDF Malware Detection: Machine Learning Modeling with Explainability](https://ieeexplore.ieee.org/document/10412055/) - IEEE Xplore
- [PDF Forensic Analysis and XMP Metadata Streams](https://www.meridiandiscovery.com/articles/pdf-forensic-analysis-xmp-metadata/) - Meridian Discovery

### Document Provenance and Creator Identification

Existing document forensics relies heavily on metadata analysis for creator identification. Current methods extract document information dictionaries and XMP metadata streams to identify creation tools, with Google Docs PDFs uniquely identified by Skia/PDF producer signatures and empty creator fields, while Word-generated PDFs contain Microsoft-specific metadata patterns. However, these approaches are vulnerable to metadata manipulation and stripping, creating the forensic gap this research addresses.

**Key References:**
- [Use Python to Determine if PDF was Generated by Google Docs](https://stackoverflow.com/questions/60876952/use-python-to-determine-if-pdf-was-generated-by-google-docs) - Stack Overflow
- [Check if PDF has been Exported from Word/Google Docs](https://stackoverflow.com/questions/17094800/check-if-pdf-has-been-exported-from-word-google-docs) - Stack Overflow
- [PDF Forensics and the Metadata Conundrum](https://pdfa.org/presentation/pdf-forensics-and-the-metadata-conundrum/) - PDF Association
- [Digital Investigation of PDF Files](https://arxiv.org/pdf/1707.05102) - arXiv

### Research Gap and Novel Contribution

The literature review confirms this project's novelty: while binary-to-image classification is well-established for malware analysis, and PDF forensics focuses on malware detection and metadata analysis, no previous research systematically applies binary visualization techniques to PDF creator identification. The combination of controlled content generation through Wikipedia articles, binary-to-image conversion, and creator classification represents an unexplored intersection of established methodologies.

### Supporting Tools and Resources

- [PDF Forensic Tool](https://github.com/xovim001/pdf-forensic) - GitHub Repository for PDF metadata extraction
- [Malware Image Detection Implementation](https://github.com/TanayBhadula/malware-image-detection) - CNN-based binary classification example
- [Best Tool for PDF Forensics](https://community.metaspike.com/t/best-tool-for-pdf-forensics/1090) - Community discussion on current limitations

This literature foundation demonstrates that while the individual components (binary visualization, PDF forensics, document provenance) have been studied, their integration for PDF creator identification through binary signatures represents a genuine contribution to the digital forensics field.

## Performance Parameters

### Current Results (September 2025)

**BREAKTHROUGH ACHIEVED**: The binary-to-image PDF classification approach has demonstrated exceptional success with initial testing:

#### 2-Class Classification (Word vs Google Docs)
- **SVM Accuracy: 97.5%** 
- **SGD Accuracy: 97.5%**
- Training time: <1 second
- Perfect precision (100%) for Google Docs detection
- Only 1 misclassification out of 40 test samples

#### 3-Class Classification (Word vs Google Docs vs Python/ReportLab)
- **SVM Accuracy: 98.3%**
- **SGD Accuracy: 100%** ⭐ **PERFECT CLASSIFICATION**
- Training time: <1 second  
- Zero classification errors with SGD classifier
- Python-generated PDFs show distinct binary signatures (mean intensity: 73.17 vs ~121 for Word/Google)

### Success Metrics
- **Classification Accuracy**: Target ≥95% across all PDF generation methods
- **Training Speed**: Models must train in <10 seconds for practical deployment
- **Robustness**: Maintain >90% accuracy across varying document lengths and content types
- **Scalability**: Performance retention when expanding to 4+ PDF generation sources
- **Real-world Validation**: Successful classification of forensic samples from actual investigations

### Research Completion Indicators
1. **Phase 1 ✅ COMPLETE**: Proof of concept with 97.5-100% accuracy on controlled dataset
2. **Phase 2**: Validation with larger datasets (1000+ samples per class)
3. **Phase 3**: Addition of 4th+ PDF generation sources with maintained accuracy
4. **Phase 4**: Testing on real-world forensic samples and varying document characteristics

## Research Approach & Tasks
Identify high level tasks and sub tasks. Develop a realistic timeline. This timeline will be reviewed weekly. Timeline is flexible and may be adjusted. Plan out at least the first four weeks of research.

### Task Planning

**CRITICAL**: With initial proof-of-concept achieving 97.5-100% accuracy, the focus now shifts to **VALIDATION AND EXPANSION** of these remarkable results. Students must develop comprehensive verification plans to ensure the robustness and generalizability of the binary-to-image PDF classification methodology.

| Task | Description | Estimated Time to Complete |
|------|-------------|---------------------------|
| **1. Scale-Up Validation** | Expand dataset to full corpus (398 samples per class) and verify performance retention | 1 Week |
| **1.1** | Train classifiers on complete Word PDF dataset (398 samples) | 2 Days |
| **1.2** | Train classifiers on complete Google Docs PDF dataset (398 samples) | 2 Days |
| **1.3** | Train classifiers on complete Python PDF dataset (90+ samples) | 1 Day |
| **1.4** | Compare full-scale results with initial 100-sample results | 2 Days |
| **2. Fourth PDF Source Integration** | Add LibreOffice/OpenOffice as 4th PDF generation source | 2 Weeks |
| **2.1** | Generate 200+ LibreOffice PDF samples using Wikipedia content | 1 Week |
| **2.2** | Convert LibreOffice PDFs to binary images | 2 Days |
| **2.3** | Train and evaluate 4-class classifier (Word/Google/Python/LibreOffice) | 3 Days |
| **2.4** | Analyze classification performance degradation with additional class | 2 Days |
| **3. Document Length Variation Testing** | Verify robustness across varying PDF document lengths | 1.5 Weeks |
| **3.1** | Create short documents (1-2 pages) across all PDF generation methods | 3 Days |
| **3.2** | Create long documents (10+ pages) across all PDF generation methods | 3 Days |
| **3.3** | Train length-stratified classifiers and compare performance | 4 Days |
| **3.4** | Analyze impact of document length on binary signature patterns | 1 Day |
| **4. Content Type Robustness** | Test classifier performance across different document content types | 2 Weeks |
| **4.1** | Generate mathematical/formula-heavy PDFs | 3 Days |
| **4.2** | Generate image-heavy PDFs | 3 Days |
| **4.3** | Generate table-heavy PDFs | 3 Days |
| **4.4** | Generate plain text-only PDFs | 2 Days |
| **4.5** | Evaluate classifier performance across content type variations | 3 Days |
| **5. Advanced Classifier Development** | Explore CNN and deep learning approaches for improved accuracy | 2 Weeks |
| **5.1** | Design CNN architecture for binary image classification | 3 Days |
| **5.2** | Implement and train CNN models | 5 Days |
| **5.3** | Compare CNN performance with baseline SVM/SGD results | 2 Days |
| **5.4** | Optimize CNN hyperparameters for maximum accuracy | 4 Days |
| **6. Real-World Validation** | Test classifiers on authentic forensic samples | 2 Weeks |
| **6.1** | Collect real-world PDF samples from various sources | 1 Week |
| **6.2** | Apply trained classifiers to real-world samples | 2 Days |
| **6.3** | Validate results against known PDF generation sources | 3 Days |
| **6.4** | Document edge cases and failure modes | 2 Days |
| **7. Adversarial Testing** | Evaluate classifier robustness against evasion attempts | 1.5 Weeks |
| **7.1** | Test performance on PDFs with stripped metadata | 2 Days |
| **7.2** | Test performance on PDF-to-PDF conversions | 3 Days |
| **7.3** | Test performance on compressed/optimized PDFs | 3 Days |
| **7.4** | Document classifier limitations and attack vectors | 3 Days |
| **8. Performance Optimization** | Optimize classifiers for deployment in forensic environments | 1 Week |
| **8.1** | Implement feature selection for faster training | 2 Days |
| **8.2** | Develop lightweight model versions for resource-constrained environments | 3 Days |
| **8.3** | Create automated pipeline for PDF-to-classification workflow | 2 Days |
| **Final Task** | Develop Final Research Paper and Presentation documenting validation results | 2 Weeks |

### Weekly Milestones
- **Week 1**: Complete scale-up validation and begin 4th source integration
- **Week 2**: Finish LibreOffice integration and begin document length testing  
- **Week 3**: Complete content type robustness testing
- **Week 4**: Begin CNN development and real-world validation
- **Week 5-6**: Complete advanced classifier development
- **Week 7-8**: Finish real-world validation and adversarial testing
- **Week 9-10**: Performance optimization and final documentation

### Experimental Validation Recommendations

#### Critical Research Questions to Address:
1. **Scalability**: Do the 97.5-100% accuracy results hold with larger datasets?
2. **Generalizability**: Can the approach distinguish between 4+ PDF generation sources?
3. **Robustness**: How sensitive are binary signatures to document characteristics (length, content type, formatting)?
4. **Real-world Applicability**: Do lab results translate to authentic forensic scenarios?
5. **Adversarial Resistance**: Can the approach detect deliberate attempts to obscure PDF provenance?

#### Recommended Experimental Controls:
- **Consistent Content**: Use identical Wikipedia articles across all PDF generation methods
- **Balanced Datasets**: Maintain equal sample sizes per class where possible
- **Stratified Testing**: Separate validation by document length, content type, and generation method
- **Blind Validation**: Test on PDFs where generation method is unknown to researchers
- **Cross-validation**: Use k-fold validation to ensure results aren't dependent on specific train/test splits

#### Success Criteria for Validation:
- **Full-scale accuracy ≥90%** across all PDF generation methods
- **4-class classification accuracy ≥85%** with addition of LibreOffice
- **Length-invariant performance**: <5% accuracy drop across short vs long documents
- **Content-type robustness**: <10% accuracy drop across different document types
- **Real-world validation**: Successfully classify ≥80% of authentic forensic samples

### Key Deliverables
- **Validation Results Report:** Due Mid-October (Week 4)
- **Advanced Classifier Implementation:** Due End of October (Week 6) 
- **Real-World Testing Results:** Due Mid-November (Week 8)
- **Final research paper draft:** Due Last Week of November
- **Final research paper:** Due First Week of December
- **Final presentation of research:** Performed Second Week December

### Research Impact Goals
This validation phase will determine whether the binary-to-image PDF classification approach can:
1. **Scale to forensic deployment** with maintained accuracy
2. **Handle diverse real-world scenarios** beyond controlled lab conditions  
3. **Resist adversarial attempts** to disguise PDF provenance
4. **Provide reliable evidence** for cybersecurity investigations and legal proceedings

---
*Template based on CYB 610: Cybersecurity Project requirements*
