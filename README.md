# Predict Customer Churn <!-- omit in toc -->

This is the first project **Predict Customer Churn** of the ML DevOps Engineer Nanodegree
from **Udacity**.

- [2. Developing](#2-developing)
  - [2.1. Setup](#21-setup)
  - [2.2. Build in functionality](#22-build-in-functionality)
  - [2.3. Dependencies](#23-dependencies)
  - [2.4. Environment variables and configuration](#24-environment-variables-and-configuration)

## 2. Developing

### 2.1. Setup

1. Install docker desktop (on Mac)

  ```bash
  brew install --cask docker
  ```

2. Build and start ide

```bash
docker-compose build ide  
docker-compose up ide
```

The same can be achieved in VS Code by running the command `Rebuild and Reopen in Container` via the
command prompt (`F1` or `CMD+Shift+P`). Requires the Dev Containers extension (see step 3 below).
This also updates changes in .devcontainer/devcontainer.json and .vscode/settings.json.

3. Attach to running container with `Dev Containers` extension

Install [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension on your host VsCode.\
Copy and modify a version of ".devcontainer/devcontainer.json.example" to your needs.\
Press `F1` or `CMD+Shift+P` and select `Dev Containers: Attach to Running Container...`.

You might have to navigate to the project root at `home/route/app.

### 2.2. Build in functionality

1. Use your host ssh credentials in the container.\
    Add AgentForwarding to ssh-config.\
    Add your key to docker started ssh-agent (important! it is not the default one) see:
    <https://stackoverflow.com/a/64023733>.

    ```bash
    SSH_AUTH_SOCK=`launchctl getenv SSH_AUTH_SOCK` ssh-add
    ```

2. Make sure that your local git config is setup such that it can be mounted into the container:

    ``` bash
    git config --global user.name "Your Name"
    git config --global user.email "youremail@yourdomain.com"
    git config --global --add safe.directory /home/root/app
    ```

    Note: If you have no git credentials on your host system this can lead to ugly mounting
    behavior. Remove the `.gitconfig` mount in `docker-compose.yaml` file, to remove this feature

### 2.3. Dependencies

**Poetry* is used as dependency manager. We don't use a virtual environment and install dependencies
directly at container level.

### 2.4. Environment variables and configuration

For the dev-containers the built in functionality of using .env files is used.\
If you want to make updates to the environment, be aware that changes only apply after container restart (`docker-compose up --force-recreate ide`) .

The priority of the configuration values is as follows:

1. Initialisation in init.py of the privacy_filter module.
2. Variables from the environment
   1. Variables set via docker-compose.yaml under environment
   2. Variables set via the .env file that is loaded in docker-compose.yaml (first from .env, then
   overwritten from .env_dev/.env_prd).
