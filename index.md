## Abstract
<!-- Theme Transformer: Symbolic Music Generation with Theme-Conditioned Transformer -->

At the core of a Transformer model lies the so-called self-attention mechanism that supplies its autoregressive generation process with long-term memory. To exert control over the generation process of the Transformer, a popular and universal approach is to "prompt" the Transformer with a user-specified sequence and ask the model to generate a continuation. When using neural networks for music generation, the self-attention mechanism contributes to the consistency of the generated pieces in harmony and style. However, we argue in this paper that the self-attention mechanism alone is not sufficient for music applications because it cannot guarantee that the prompt would repeat itself or manifest in some way in the generated continuation. To improve this shortcoming, we propose an alternative conditioning approach, named theme-based conditioning, that explicitly coaches the Transformer to have multiple occurrences of the given condition, or a ``theme,'' in its generation. Technically, this is achieved by associating a theme with its occurrences in a training piece by contrastive representation learning and clustering, and by establishing a separate and dedicated memory network for the conditioning theme, so that the Transformer can self-attend to the long-term memory and cross-attend to the theme in parallel. We report on objective and subjective evaluations of variants of the proposed Theme Transformer and the conventional prompt-based baseline, showing that our best model can generate, to some extent, polyphonic pop piano music with repetition and plausible variations of the given condition. 

## Demo
| ID| Theme | Real Data | Baseline | Theme Transformer
| -- | -------- | -------- | -------- | -------- |
| 875 | `audio: /theme-transformer-audio/875_Theme.mp3` | `audio: /theme-transformer-audio/875_Realdata.mp3` | `audio: /theme-transformer-audio/875_Baseline.mp3` | `audio: /theme-transformer-audio/875_ThemeTransformer.mp3`|
| 888 | `audio: /theme-transformer-audio/888_Theme.mp3` | `audio: /theme-transformer-audio/888_Realdata.mp3` | `audio: /theme-transformer-audio/888_Baseline.mp3` | `audio: /theme-transformer-audio/888_ThemeTransformer.mp3`|
| 890 | `audio: /theme-transformer-audio/890_Theme.mp3` | `audio: /theme-transformer-audio/890_Realdata.mp3` | `audio: /theme-transformer-audio/890_Baseline.mp3` | `audio: /theme-transformer-audio/890_ThemeTransformer.mp3`|
| 893 | `audio: /theme-transformer-audio/893_Theme.mp3` | `audio: /theme-transformer-audio/893_Realdata.mp3` | `audio: /theme-transformer-audio/893_Baseline.mp3` | `audio: /theme-transformer-audio/893_ThemeTransformer.mp3`|
| 899 | `audio: /theme-transformer-audio/899_Theme.mp3` | `audio: /theme-transformer-audio/899_Realdata.mp3` | `audio: /theme-transformer-audio/899_Baseline.mp3` | `audio: /theme-transformer-audio/899_ThemeTransformer.mp3`|
| 900 | `audio: /theme-transformer-audio/900_Theme.mp3` | `audio: /theme-transformer-audio/900_Realdata.mp3` | `audio: /theme-transformer-audio/900_Baseline.mp3` | `audio: /theme-transformer-audio/900_ThemeTransformer.mp3`|
| 901 | `audio: /theme-transformer-audio/901_Theme.mp3` | `audio: /theme-transformer-audio/901_Realdata.mp3` | `audio: /theme-transformer-audio/901_Baseline.mp3` | `audio: /theme-transformer-audio/901_ThemeTransformer.mp3`|
| 904 | `audio: /theme-transformer-audio/904_Theme.mp3` | `audio: /theme-transformer-audio/904_Realdata.mp3` | `audio: /theme-transformer-audio/904_Baseline.mp3` | `audio: /theme-transformer-audio/904_ThemeTransformer.mp3`|
| 908 | `audio: /theme-transformer-audio/908_Theme.mp3` | `audio: /theme-transformer-audio/908_Realdata.mp3` | `audio: /theme-transformer-audio/908_Baseline.mp3` | `audio: /theme-transformer-audio/908_ThemeTransformer.mp3`|
| 909 | `audio: /theme-transformer-audio/909_Theme.mp3` | `audio: /theme-transformer-audio/909_Realdata.mp3` | `audio: /theme-transformer-audio/909_Baseline.mp3` | `audio: /theme-transformer-audio/909_ThemeTransformer.mp3`|


## Figures
|#id| First 24 bars  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   | Melody Embedding Distance|
| ------ | -------- | -------- | 
|875| <img src="testdata_24bars_modelC/test_875_front24.jpg" width="100%"/> |     | 
|888| <img src="testdata_24bars_modelC/test_888_front24.jpg" width="100%"/> |     | 
|890| <img src="testdata_24bars_modelC/test_890_front24.jpg" width="100%"/> |     | 
|893| <img src="testdata_24bars_modelC/test_893_front24.jpg" width="100%"/> |     | 
|894| <img src="testdata_24bars_modelC/test_894_front24.jpg" width="100%"/> |     | 
|896| <img src="testdata_24bars_modelC/test_896_front24.jpg" width="100%"/> |     | 
|899| <img src="testdata_24bars_modelC/test_899_front24.jpg" width="100%"/> |     | 
|900| <img src="testdata_24bars_modelC/test_900_front24.jpg" width="100%"/> |  | 
|901| <img src="testdata_24bars_modelC/test_901_front24.jpg" width="100%"/> |     | 
|904| <img src="testdata_24bars_modelC/test_904_front24.jpg" width="100%"/> |     | 
|908| <img src="testdata_24bars_modelC/test_908_front24.jpg" width="100%"/> |     | 
|909| <img src="testdata_24bars_modelC/test_909_front24.jpg" width="100%"/> |     | 

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/atosystem/midi2Tiles/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.