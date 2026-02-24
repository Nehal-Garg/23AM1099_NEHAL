pipeline {
    agent any

    stages {

        stage('GitHub Access') {
            steps {
                echo 'Cloning Repository...'
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                echo 'Upgrading pip...'
                bat 'python -m pip install --upgrade pip'

                echo 'Installing dependencies from requirement.txt...'
                bat 'pip install -r exp2\\requirement.txt'
            }
        }

        stage('Training Stage') {
            steps {
                echo 'Running Training Script...'
                bat 'python exp2\\train.py'
            }
        }
    }
}
