# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Node.js and PostgreSQL development container project. The project uses VS Code Dev Containers for a consistent development environment with Node.js 22 and PostgreSQL database.

## Development Environment

- **Runtime**: Node.js 22 (Bookworm)
- **Database**: PostgreSQL (latest)
- **Container**: Docker/Docker Compose
- **IDE**: VS Code with Dev Containers

## Database Connection

PostgreSQL is available at:
- **Host**: `db` (from within the app container)
- **Port**: 5432
- **Database**: `postgres`
- **Username**: `postgres`
- **Password**: `postgres`

## Common Development Tasks

The ANP (Adversarial Noise Propagation) web application is now fully implemented. Here are the main commands:

```bash
# Start development server
npm run dev

# Start production server  
npm start

# Initialize database schema
node src/utils/initDB.js

# Seed sample data
node src/utils/seedData.js
```

## Project Structure

The application implements the ANP algorithm from the research paper:
- `src/anp/` - ANP algorithm implementation and related utilities
- `src/routes/` - API endpoints for training, evaluation, and metrics
- `src/utils/` - Database utilities and initialization scripts
- `public/` - Frontend assets (CSS, JavaScript, images)
- `views/` - EJS templates for the web interface
- `uploads/` - Model file uploads directory

## Features Implemented

### Core ANP Algorithm
- **Layer-wise noise injection**: Injects adversarial noise into hidden layers during training
- **Progressive noise propagation**: K-step gradient descent for noise computation
- **Shallow layer focus**: Prioritizes top-k layers as shown to be most critical
- **Multiple attack methods**: FGSM, BIM, PGD, MI-FGSM implementations

### Web Interface
- **Training Dashboard**: Configure and monitor ANP training with real-time charts
- **Model Evaluation**: Test robustness against various attacks and corruptions
- **Attack Visualization**: Generate and visualize adversarial examples
- **Metrics Comparison**: Compare ANP vs standard models with comprehensive metrics
- **Leaderboard**: Rank models by robustness performance

### Database Schema
- Models and training history tracking
- Robustness test results storage
- User management
- Model weights storage capability

## Environment Configuration

Database connection is pre-configured for the dev container:
- **Host**: db (PostgreSQL container)
- **Port**: 5432
- **Database**: postgres
- **Credentials**: postgres/postgres

The application runs on http://localhost:3000 with sample data pre-loaded including ANP-trained and standard models for comparison.