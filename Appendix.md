# Appendix

## Extracting Features

The following table lists the feature extraction and similarity metrics:

**Table: Feature extraction and similarity metrics for APK analysis**

<table>
<thead>
<tr>
<th align="center">No.</th>
<th align="left">Feature</th>
<th align="left">Representation and Similarity Metric</th>
<th align="center">Effective</th>
</tr>
</thead>
<tbody>
<tr><td colspan="4" align="left"><strong>Metadata & Resources</strong></td></tr>
<tr><td align="center">1</td><td>APK Name</td><td>Character-level lexical similarity using Levenshtein and Jaro-Winkler distances</td><td align="center">✗</td></tr>
<tr><td align="center">2</td><td>Resource Files</td><td>Jaccard similarity over sets of resource directories (e.g., <code>drawable</code>, <code>layout</code>)</td><td align="center">✗</td></tr>
<tr><td align="center">3</td><td>Manifest Labels</td><td>Jaccard similarity over application labels and declared permissions</td><td align="center">✗</td></tr>
<tr><td colspan="4" align="left"><strong>Visual Appearance</strong></td></tr>
<tr><td align="center">4</td><td>Icon Content</td><td>Deep visual content embeddings extracted by a CNN, compared using cosine similarity</td><td align="center">✓</td></tr>
<tr><td align="center">5</td><td>Icon Style</td><td>Deep visual style embeddings extracted from intermediate CNN layers, compared using cosine similarity</td><td align="center">✓</td></tr>
<tr><td colspan="4" align="left"><strong>Smali Bytecode</strong></td></tr>
<tr><td align="center">6</td><td>Smali File Count</td><td>Euclidean distance over the total number of decompiled Smali files</td><td align="center">✗</td></tr>
<tr><td align="center">7</td><td>Smali File Size</td><td>Euclidean distance over the aggregated size of Smali bytecode</td><td align="center">✗</td></tr>
<tr><td align="center">8</td><td>Smali Instr. Freq.</td><td>Jaccard similarity over opcode frequency distributions</td><td align="center">✗</td></tr>
<tr><td align="center">9</td><td>Smali Transition</td><td>Cosine similarity over opcode transition probability matrices</td><td align="center">✓</td></tr>
<tr><td colspan="4" align="left"><strong>Native Code (Shared Objects)</strong></td></tr>
<tr><td align="center">10</td><td>SO Lib Distribution</td><td>Jaccard similarity over native library counts across CPU architectures</td><td align="center">✗</td></tr>
<tr><td align="center">11</td><td>SO Instr. Freq.</td><td>Jaccard similarity over native instruction frequency distributions</td><td align="center">✗</td></tr>
<tr><td align="center">12</td><td>SO Transition</td><td>Cosine similarity over native instruction transition matrices</td><td align="center">✓</td></tr>
<tr><td colspan="4" align="left"><strong>Semantic & API</strong></td></tr>
<tr><td align="center">13</td><td>API Frequency</td><td>Jaccard similarity over API invocation frequency sets</td><td align="center">✗</td></tr>
<tr><td align="center">14</td><td>API Call Graph (GNN)</td><td>Graph Neural Network-based embeddings of API call graphs (Cosine similarity)</td><td align="center">✗</td></tr>
<tr><td align="center">15</td><td>API Call Graph (OT)</td><td>Optimal Transport distance between attributed API call graphs</td><td align="center">✓</td></tr>
</tbody>
</table>

**Abbreviations:**
- CNN: Convolutional Neural Network
- GNN: Graph Neural Network
- OT: Optimal Transport
- SO: Shared Object (Native Library)
- Instr.: Instruction
- Freq.: Frequency

## Details of Implementation

### Package Unpacking

For the unpacking phase, we employ platform-specific tools to decompile application packages into intermediate representations:

1. **Android Applications**: We leverage Apktool to decompile APK files into their corresponding smali representations. To reduce noise introduced by third-party libraries, we apply a maintained list of known third-party components to filter the smali files based on their relative paths. This allows us to retain only the application-specific code for further analysis.

2. **HarmonyOS Applications**: For Harmony Ability Packages, we utilize a static analysis tool provided by our collaborator to extract ArkCompiler Bytecode (ABC) files from HAP packages. Similar to the Android processing pipeline, we employ an SDK list to filter out common library components, preserving only the core functional code opcodes for subsequent similarity analysis. This consistent approach ensures comparable code representations across both platforms while accounting for their distinct package formats and bytecode specifications.

### TPL Collection

To construct a robust code fingerprint library, we adopt a two-stage collection strategy to identify and exclude non-unique code. Specifically, we collect both framework-specific common libraries and high-frequency third-party libraries (TPLs) from Android and HarmonyOS ecosystems.

#### 1. Framework-specific Common Library Collection

We first identify the development framework used by each application in order to isolate framework-inherent code. Cross-platform frameworks (e.g., Flutter, React Native, and Unity) encapsulate their core runtime logic in native C/C++ engines, which are distributed as shared libraries. Therefore, we leverage distinctive `.so` filenames as framework signatures. Based on a curated list of 12 representative frameworks and their corresponding `.so` identifiers (summarized in Table 3), we classify all collected applications into framework-specific categories.

For each framework category, we randomly sample up to 100 representative applications. Their APKs are unpacked, and DEX files are converted into Java archives (JAR) using `dex2jar`. We then parse all non-system package namespaces, excluding standard libraries such as `android.*` and `java.*`. By analyzing the occurrence frequency and cross-app distribution of these packages, we identify libraries that consistently appear across applications built on the same framework. To further remove generic or utility libraries, we employ a Large Language Model (LLM) as an auxiliary filter, retaining only framework-representative components. Through this process, we finally identify **836 framework-specific common libraries** associated with the 12 frameworks.

#### 2. Popular TPL Collection

Beyond framework-inherent components, we further collect widely used third-party libraries to eliminate common functional code. For **Android**, we crawl the Maven Central Repository based on "Popular Categories" and "Popular Tags". Libraries with a usage count exceeding 50 are retained, resulting in **1,347 high-frequency Android TPLs**. For **HarmonyOS**, we collect third-party libraries from the OpenHarmony Community Repository, yielding **2,391 widely adopted HarmonyOS TPLs** that constitute the main dependency ecosystem.

#### 3. Preprocessing and Filtering

In total, we construct a blocklist containing **836 framework-specific common libraries** and **3,738 high-frequency TPLs**, amounting to **4,574 libraries**. During data preprocessing, each application is decompiled (using Apktool for Android), and all Smali files are extracted. Any file corresponding to a library in the blocklist is removed. The remaining code, which primarily represents application-specific business logic, is then forwarded to the subsequent feature extraction module.

### Icon Extraction

Application icons serve as the primary visual identifiers of Android apps and provide valuable cues for detecting repackaging or imitation. Since icons are stored as resources within the APK file, it is necessary to locate and extract them during preprocessing. The rationale behind icon extraction is that counterfeit or rebranded apps often attempt to mimic the icons of popular applications, making icon similarity an important signal for detecting unauthorized republishing.

The icon extraction process follows these steps:

1. Decompile each APK using appropriate tools
2. Parse the associated XML configuration files (e.g., AndroidManifest.xml and resources.arsc), which specify the file path of the application's launcher icon
3. Locate the corresponding image files in the resource directory (typically under `res/mipmap-*/` or `res/drawable-*/`)
4. Extract and store the referenced icon images in a dedicated directory for subsequent analysis

This procedure ensures that each application is paired with its official launcher icon, which can later be leveraged for visual similarity comparison and feature extraction.

### Native Library Opcode Extraction

In addition to Java bytecode, Android applications frequently incorporate native code packaged as shared object (.so) libraries. These native binaries often implement performance-critical or security-sensitive functionality and therefore provide valuable features for program analysis. Similar to the opcode-based analysis of Smali code, we represent the behavior of native libraries through opcode transition features. The key motivation is that the instruction-level control-flow patterns of native code can serve as discriminative signals for identifying framework usage or malicious behaviors.

The feature extraction process consists of the following steps:

1. **Disassembly**: Disassemble each .so file into its assembly instructions
2. **Vocabulary Construction**: Unlike Smali code, native binaries vary across processor architectures (e.g., ARM, ARM64, x86, MIPS), leading to multiple distinct opcode sets. To address this challenge, we conducted a statistical analysis over the large-scale AndroZoo dataset and constructed a unified vocabulary consisting of the top 94 most frequent opcodes observed across architectures
3. **Transition Matrix Computation**: For each method, we performed a linear scan over its instruction sequence and computed opcode transition probabilities by counting how often an opcode a is immediately followed by another opcode b. The result is a 94 × 94 transition matrix, which compactly encodes the local control-flow behavior of the native library and serves as the basis for downstream feature extraction

### Opcode and Call Graph Extraction

For the extraction of static features, we used a set of custom Python scripts to analyze the cleaned smali files obtained after third-party library removal.

#### 1. API Call Graph Construction

To construct the API call graph, we:

1. Parse all `invoke-*` instructions in the smali code
2. Extract the implicit caller-callee relationship between two API methods from each instruction
3. Extract all method-level invocation pairs and organize them into directed edges
4. Aggregate these edges into a global graph structure representing the application's internal API invocation topology

#### 2. Opcode-based Features

For opcode-based features, we:

1. Define a fixed vocabulary of 256 distinct opcodes
2. Using a linear scan over each method's instruction sequence, compute opcode transition probabilities by counting how often a given opcode a is immediately followed by another opcode b
3. Generate a 256 × 256 transition matrix, where each entry captures the likelihood of sequential transitions between two instructions. This matrix serves as a compact representation of the application's local control-flow behavior

### API Crawling and Collection

We extract attribute information from API documentation. Specifically, for the documentation collection, we developed Python-based crawlers using the requests and BeautifulSoup libraries. The crawler started from the top-level index of the official Android developer site, extracted API category entries, and recursively traversed second- and third-level pages to collect fine-grained descriptions of each API. The collected API metadata, including class names, method signatures, and brief functional descriptions, were organized and stored in structured JSON format.

To enable embedding construction for Android APIs, we first needed to gather a comprehensive set of official documentation describing the functionality and usage of each API. The rationale behind this step is that APIs are not only identified by their method names, but also by their associated attributes such as parameters, return types, permissions, and exceptions. These attributes collectively capture the semantic meaning of APIs and provide essential contextual information for representation learning. Therefore, building a reliable corpus of API documentation serves as the foundation for generating high-quality API embeddings.

Specifically, we developed Python-based crawlers using the requests and BeautifulSoup libraries to automatically extract metadata from the Android developer website. The crawler started from the top-level API index page, where it collected all primary package names. For each package, the crawler accessed its dedicated page to gather detailed API entries, including method names, functional descriptions, input parameters, output return values, required permissions, and possible exceptions. When intermediate pages contained only category information instead of concrete API methods, the crawler continued recursive traversal of second- and third-level links until all relevant APIs were enumerated. All extracted metadata was then normalized and stored in structured JSON format, which provides the input for the subsequent API embedding process.

### API Embedding

After collecting structured metadata for Android APIs, we construct semantic embeddings to enhance the expressiveness of the API graph. The motivation is that the raw structural graph, built solely from package hierarchies or co-occurrence, lacks the ability to represent the functional semantics of individual APIs. Attributes such as method descriptions, input/output parameters, permissions, and exceptions provide valuable contextual information, but they must be encoded into dense representations to be integrated effectively. Embedding construction therefore serves to endow each API node with semantic meaning, enabling the API graph to jointly capture structural relations and functional semantics.

To achieve this, we adopt the TransH model, a knowledge-graph-based embedding method that enables more accurate modeling of complex relations. In our formulation, API methods are treated as entities, and their connections (e.g., "belongs-to-class," "throws-exception," "requires-permission") are represented as relations. TransH introduces a relation-specific hyperplane for each relation type: both the head and tail entities are projected onto this hyperplane, after which a translation operation is performed within the subspace. This design effectively reduces conflicts among heterogeneous relations and allows fine-grained distinctions between different relation semantics. As a result, each API is represented by a fixed-dimensional semantic embedding that reflects not only its textual attributes but also its contextual relationships, thereby enhancing the semantic richness of the API graph.

### Implementation Details of Baseline Methods

Since none of these systems provide public implementations, we re-implement their core components following the algorithmic descriptions in their original papers. To ensure fairness and reproducibility, all parameters are tuned on the validation split of D1 unless explicit recommendations are available.

- **GeminiScope**: We reconstruct the control-flow and program-dependence graph extraction pipeline using `FlowDroid` and `Soot`. The similarity score follows the node-edge matching formulation described in the original paper, combined with a cosine-based structural similarity metric. The clone decision threshold is selected via grid search on D1.

- **SUIDroid**: UI hierarchies are obtained using `uiautomator` dumps. View-tree similarity is computed with the structure-preserving edit-distance model proposed by Lyu et al. We adopt their recommended matching costs and depth limits, while tuning the decision threshold on D1.

- **FSquaDRA**: Resource fingerprints are generated from manifests, XML layouts, and drawable assets. We follow the original design by using perceptual hashing (`pHash`) and the resource-overlap coefficient for similarity scoring. The threshold is again calibrated using D1.

## Supplementary Materials

### APK File Structure

![APK File Structure](images/apk.png)

*Overview of the decompiled APK file structure and the corresponding critical resources extracted for feature engineering. Key directories include `res` for visual assets, `lib` for native code, and `smali` for Smali code.*

### Code Obfuscation Techniques

The following table lists the code obfuscation techniques:

**Table: Taxonomy of Contemporary Code Obfuscation Techniques**

<table>
<thead>
<tr>
<th align="left">Obfuscation Category</th>
<th align="left">Description</th>
<th align="left">Representative Techniques</th>
</tr>
</thead>
<tbody>
<tr>
<td>Identifier Renaming</td>
<td>Substitutes semantically meaningful identifiers (e.g., <code>calculateInterest()</code>) with nonsensical or opaque tokens, severing symbolic associations to hinder program comprehension.</td>
<td>ClassRename, FieldRename, MethodRename</td>
</tr>
<tr>
<td>Control and Data Flow Disruption</td>
<td>Obfuscates control flow through non-deterministic execution paths (e.g., opaque predicates, trampoline jumps) and obscures data flow via variable splitting and redundant state manipulation.</td>
<td>AdvancedReflection, ArithmeticBranch, CallIndirection, DebugRemoval</td>
</tr>
<tr>
<td>Cryptographic Masking</td>
<td>Protects critical assets—such as strings, libraries, and resource files—via encryption, coupled with runtime decryption stubs to thwart static analysis.</td>
<td>AssetEncryption, ConstStringEncryption, LibEncryption, ResStringEncryption</td>
</tr>
<tr>
<td>Syntactic Trivialization</td>
<td>Introduces semantically inert transformations (e.g., dead code, no-op instructions, randomized metadata) to degrade the effectiveness of pattern-based reverse engineering tools.</td>
<td>NewAlignment, NewSignature</td>
</tr>
<tr>
<td>Resource Entanglement</td>
<td>Obfuscates application resource files (e.g., <code>AndroidManifest.xml</code>, <code>strings.xml</code>) through path randomization, checksum spoofing, and compression, disrupting the logical-physical mapping.</td>
<td>Path randomization, checksum spoofing, resource compression</td>
</tr>
</tbody>
</table>

### Graph Similarity

![Graph Similarity Comparison](images/model_vs_ot_scatter.svg)

*Comparison of OT-based and GNN-based graph similarity scoring. The OT-based method maintains clearer separation between positive and negative pairs.*

### Obfuscation Robustness Result

To evaluate the robustness of our approach against code obfuscation techniques, we conducted comprehensive experiments on both Android and HarmonyOS platforms. We tested the identification performance under various single obfuscation techniques as well as their combinations.

#### HarmonyOS Obfuscation Results

For HarmonyOS applications, we evaluated four categories of obfuscation techniques: **Basic** (identifier renaming), **Data** (data flow obfuscation), **Inst** (instruction-level transformations), and **Ctrl** (control flow obfuscation). The following table presents the identification performance:

**Table: Obfuscation identification performance on HarmonyOS**

<table>
<thead>
<tr>
<th align="center">Obfuscation Type</th>
<th align="center">Basic</th>
<th align="center">Data</th>
<th align="center">Inst</th>
<th align="center">Ctrl</th>
<th align="center">Precision</th>
<th align="center">Recall</th>
</tr>
</thead>
<tbody>
<tr><td colspan="7" align="left"><strong>Single Obfuscation</strong></td></tr>
<tr><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">100.00%</td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td></td><td></td><td align="center">85.00%</td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td align="center">✓</td><td></td><td align="center">100.00%</td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td align="center">✓</td><td align="center">75.00%</td><td align="center">88.00%</td></tr>
<tr><td colspan="7" align="left"><strong>Combination Obfuscation</strong></td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td align="center">94.00%</td><td align="center">100.00%</td></tr>
<tr><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td></td><td align="center">83.00%</td><td align="center">88.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">100.00%</td><td align="center">94.00%</td></tr>
<tr><td></td><td align="center">✓</td><td></td><td></td><td align="center">✓</td><td align="center">93.00%</td><td align="center">82.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">78.00%</td><td align="center">82.00%</td></tr>
<tr><td></td><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">84.00%</td><td align="center">94.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">76.00%</td><td align="center">94.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">68.00%</td><td align="center">82.00%</td></tr>
<tr><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">78.00%</td><td align="center">82.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td align="center">100.00%</td><td align="center">88.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td align="center">68.00%</td><td align="center">88.00%</td></tr>
</tbody>
</table>

The results demonstrate that our approach maintains high identification accuracy even under single obfuscation techniques, achieving 100% recall for most cases. For combination obfuscation scenarios, the performance remains robust with recall rates ranging from 82% to 100%, indicating strong resilience against complex obfuscation strategies.

#### Android Obfuscation Results

For Android applications, we evaluated 15 distinct obfuscation techniques covering control flow manipulation, reflection, instruction reordering, and various encryption methods. The following table shows the recall performance:

**Table: Android obfuscation identification performance**

<table>
<thead>
<tr>
<th align="left">Obfuscation Type</th>
<th align="center">GOTO</th>
<th align="center">REF</th>
<th align="center">RE-ORD</th>
<th align="center">NOP</th>
<th align="center">FLD-RN</th>
<th align="center">STR-ENC</th>
<th align="center">DBG-RMV</th>
<th align="center">ARITH</th>
<th align="center">RES-ENC</th>
<th align="center">MTH-OV</th>
<th align="center">CLS-RN</th>
<th align="center">ADV-REF</th>
<th align="center">MAN-RND</th>
<th align="center">CALL-IND</th>
<th align="center">LIB-ENC</th>
<th align="center">Recall</th>
</tr>
</thead>
<tbody>
<tr><td colspan="17" align="left"><strong>Single Obfuscation</strong></td></tr>
<tr><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">95.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td align="center">95.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">75.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td align="center">100.00%</td></tr>
<tr><td colspan="17" align="left"><strong>Combination Obfuscation</strong></td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">86.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td align="center">88.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">89.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td align="center">86.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">89.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td></td><td></td><td align="center">✓</td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td align="center">✓</td><td></td><td align="center">89.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td></td><td></td><td align="center">88.00%</td></tr>
<tr><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">89.00%</td></tr>
<tr><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td></td><td align="center">100.00%</td></tr>
<tr><td></td><td align="center">✓</td><td></td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">✓</td><td></td><td></td><td></td><td align="center">✓</td><td align="center">✓</td><td></td><td align="center">✓</td><td></td><td align="center">86.00%</td></tr>
</tbody>
</table>
**Abbreviations:**

- **GOTO**: Control Flow (goto injection)
- **REF**: Reflection
- **RE-ORD**: Instruction Reordering
- **NOP**: No-Operation insertion
- **FLD-RN**: Field Renaming
- **STR-ENC**: String Encryption
- **DBG-RMV**: Debug Information Removal
- **ARITH**: Arithmetic Branch obfuscation
- **RES-ENC**: Resource Encryption
- **MTH-OV**: Method Overloading
- **CLS-RN**: Class Renaming
- **ADV-REF**: Advanced Reflection
- **MAN-RND**: Manifest Randomization
- **CALL-IND**: Call Indirection
- **LIB-ENC**: Library Encryption

The Android obfuscation results show that our approach achieves excellent recall rates (≥95%) for most single obfuscation techniques. Even under complex combination obfuscation scenarios involving up to 8 different techniques simultaneously, the system maintains recall rates between 86% and 100%, demonstrating strong robustness against sophisticated obfuscation strategies commonly employed in real-world malicious applications.

