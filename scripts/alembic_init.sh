#!/usr/bin/env bash
# Inicializa alembic e cria a primeira migration a partir dos modelos
set -e
alembic init database/migrations
alembic revision --autogenerate -m "initial_schema"
alembic upgrade head
