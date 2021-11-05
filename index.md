## Abstract
<!-- Theme Transformer: Symbolic Music Generation with Theme-Conditioned Transformer -->

At the core of a Transformer model lies the so-called self-attention mechanism that supplies its autoregressive generation process with long-term memory. To exert control over the generation process of the Transformer, a popular and universal approach is to "prompt" the Transformer with a user-specified sequence and ask the model to generate a continuation. When using neural networks for music generation, the self-attention mechanism contributes to the consistency of the generated pieces in harmony and style. However, we argue in this paper that the self-attention mechanism alone is not sufficient for music applications because it cannot guarantee that the prompt would repeat itself or manifest in some way in the generated continuation. To improve this shortcoming, we propose an alternative conditioning approach, named theme-based conditioning, that explicitly coaches the Transformer to have multiple occurrences of the given condition, or a ``theme,'' in its generation. Technically, this is achieved by associating a theme with its occurrences in a training piece by contrastive representation learning and clustering, and by establishing a separate and dedicated memory network for the conditioning theme, so that the Transformer can self-attend to the long-term memory and cross-attend to the theme in parallel. We report on objective and subjective evaluations of variants of the proposed Theme Transformer and the conventional prompt-based baseline, showing that our best model can generate, to some extent, polyphonic pop piano music with repetition and plausible variations of the given condition. 

## Demo
<audio controls>
  <source src="theme-transformer-audio/875_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>

| ID| Theme | Real Data | Baseline | Theme Transformer
| -- | -------- | -------- | -------- | -------- |
| 875 | <audio controls>
  <source src="theme-transformer-audio/875_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/875_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/875_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/875_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 888 | <audio controls>
  <source src="theme-transformer-audio/888_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/888_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/888_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/888_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 890 | <audio controls>
  <source src="theme-transformer-audio/890_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/890_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/890_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/890_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 893 | <audio controls>
  <source src="theme-transformer-audio/893_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/893_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/893_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/893_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 899 | <audio controls>
  <source src="theme-transformer-audio/899_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/899_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/899_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/899_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 900 | <audio controls>
  <source src="theme-transformer-audio/900_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/900_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/900_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/900_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 901 | <audio controls>
  <source src="theme-transformer-audio/901_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/901_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/901_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/901_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 904 | <audio controls>
  <source src="theme-transformer-audio/904_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/904_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/904_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/904_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 908 | <audio controls>
  <source src="theme-transformer-audio/908_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/908_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/908_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/908_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|
| 909 | <audio controls>
  <source src="theme-transformer-audio/909_Theme.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/909_Realdata.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/909_Baseline.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> | <audio controls>
  <source src="theme-transformer-audio/909_ThemeTransformer.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio>|


## Figures
|#id| First 24 bars | Melody Embedding Distance|
| ------ | -------- | -------- | 
|875| ![875_front_24](testdata_24bars_modelC/test_875_front24.jpg) | ![875_front_24](testdata_24bars_modelC/test_875_front24.jpg) | 
|888| ![888_front_24](testdata_24bars_modelC/test_888_front24.jpg) | ![888_front_24](testdata_24bars_modelC/test_888_front24.jpg) | 
|890| ![890_front_24](testdata_24bars_modelC/test_890_front24.jpg) | ![890_front_24](testdata_24bars_modelC/test_890_front24.jpg) | 
|893| ![893_front_24](testdata_24bars_modelC/test_893_front24.jpg) | ![893_front_24](testdata_24bars_modelC/test_893_front24.jpg) | 
|894| ![894_front_24](testdata_24bars_modelC/test_894_front24.jpg) | ![894_front_24](testdata_24bars_modelC/test_894_front24.jpg) | 
|896| ![896_front_24](testdata_24bars_modelC/test_896_front24.jpg) | ![896_front_24](testdata_24bars_modelC/test_896_front24.jpg) | 
|899| ![899_front_24](testdata_24bars_modelC/test_899_front24.jpg) | ![899_front_24](testdata_24bars_modelC/test_899_front24.jpg) | 
|900| ![900_front_24](testdata_24bars_modelC/test_900_front24.jpg) | ![900_front_24](testdata_24bars_modelC/test_900_front24.jpg) | 
|901| ![901_front_24](testdata_24bars_modelC/test_901_front24.jpg) | ![901_front_24](testdata_24bars_modelC/test_901_front24.jpg) | 
|904| ![904_front_24](testdata_24bars_modelC/test_904_front24.jpg) | ![904_front_24](testdata_24bars_modelC/test_904_front24.jpg) | 
|908| ![908_front_24](testdata_24bars_modelC/test_908_front24.jpg) | ![908_front_24](testdata_24bars_modelC/test_908_front24.jpg) | 
|909| ![909_front_24](testdata_24bars_modelC/test_909_front24.jpg) | ![909_front_24](testdata_24bars_modelC/test_909_front24.jpg) | 

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