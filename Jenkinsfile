pipeline {
    agent any

    stages {

        stage('GitHub Access') {
            steps {
                echo 'Cloning repository...'
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                echo 'Setting up environment...'
                bat 'python --version'
                bat 'pip install -r exp2\\requirement.txt'
            }
        }

        stage('Training Stage') {
            steps {
                echo 'Running training...'
                bat 'python exp2\\train.py'
            }
        }
    }
}
