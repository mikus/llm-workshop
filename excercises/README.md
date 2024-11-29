# Excercises for LLM workshop with LangChain JS

## How to run examples

### Prerequisities

1. NodeJS installed locally
2. OpeanAI API key set in `.env` file - copy `.env.example` to `.env` and put there gnerated [key for OpenAI API](https://platform.openai.com/settings/organization/api-keys).

### Jupyter notebooks

You can find additional explanations and guidance in jupyter notebooks (`labXX_yyyyy.nnb`).

The recommended way to use them is to open in [VisualStudio Code](https://code.visualstudio.com/) with installed dedicated extension - [Node.js notebooks](https://marketplace.visualstudio.com/items?itemName=donjayamanne.typescript-notebook).

Alternatively you can setup the [necessary environment](https://github.com/n-riesco/ijavascript) using the following steps:

#### With pip
```
pip install --upgrade pyzmq jupyter
npm install -g ijavascript
ijsinstall
```

#### With conda
```
conda install -c conda-forge jupyter
conda install nodejs
npm install -g ijavascript
ijsinstall
```