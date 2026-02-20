FROM python:3.11-slim

# Turns off writing .pyc files; superfluous on an ephemeral container.
ENV PYTHONDONTWRITEBYTECODE=1
# Seems to speed things up and makes sure logs are not lost in buffer when app crashes
ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# Create a non-root user (vscode) with a stable UID/GID like 1000:1000
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
# Install system dependencies
# sudo: change permissions for mounted files in postCreateCommand, postAttachCommand
# locales: set locale to en_US.UTF-8 otherwise we receive git warnings
# ripgrep: faster grep alternative Codex CLI uses


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sudo \
    locales \
    curl \
    git \
    procps \
    ripgrep \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8



RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "poetry==2.3.2"
#&& poetry config virtualenvs.in-project true


# Create group and user if they do not exist
RUN if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

# Workspace folder ownership (so you do not need chown in postCreateCommand)
RUN mkdir -p /workspaces \
    && chown -R "${USER_UID}:${USER_GID}" /workspaces

# Install micromamba (lightweight conda)
# ENV MAMBA_ROOT_PREFIX=/opt/micromamba
# ENV MAMBA_EXE=/usr/local/bin/micromamba

# RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
#    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
#    && mkdir -p /opt/micromamba \
#    && micromamba shell init -s bash -p /opt/micromamba

# RUN chown -R ${USERNAME}:${USERNAME} /opt/micromamba

WORKDIR /workspaces
USER ${USERNAME}
