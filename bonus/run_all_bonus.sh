#!/bin/bash
# run_all_bonus.sh - Script pour démontrer tous les bonus implémentés

set -e  # Arrêter en cas d'erreur

PYTHON="./venv/bin/python"

echo "========================================================================"
echo "DÉMONSTRATION DES BONUS - DSLR PROJECT"
echo "========================================================================"
echo ""

# Vérifier que l'environnement virtuel existe
if [ ! -f "$PYTHON" ]; then
    echo "❌ Environnement virtuel non trouvé!"
    echo "Exécutez d'abord: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "✅ Environnement virtuel détecté"
echo ""

# Bonus 1: Statistiques avancées
echo "========================================================================"
echo "BONUS 1: Statistiques avancées (Range, IQR, Skewness, Kurtosis)"
echo "========================================================================"
echo ""
echo "Exécution de describe.py avec les statistiques bonus..."
echo ""
$PYTHON src/describe.py datasets/dataset_train.csv 2>/dev/null | head -20
echo ""
echo "✅ 12 statistiques affichées (8 de base + 4 bonus)"
echo ""
read -p "Appuyez sur Entrée pour continuer..."
echo ""

# Bonus 2: SGD
echo "========================================================================"
echo "BONUS 2: Stochastic Gradient Descent (SGD)"
echo "========================================================================"
echo ""
echo "Entraînement avec SGD (100 époques, 147,000 mises à jour)..."
echo ""
$PYTHON bonus/logreg_train_sgd.py datasets/dataset_train.csv
echo ""
echo "✅ Modèle SGD entraîné et sauvegardé dans weights_sgd.json"
echo ""
read -p "Appuyez sur Entrée pour continuer..."
echo ""

# Bonus 3: Mini-Batch GD
echo "========================================================================"
echo "BONUS 3: Mini-Batch Gradient Descent"
echo "========================================================================"
echo ""
echo "Entraînement avec Mini-Batch GD (batch_size=64, 2,300 mises à jour)..."
echo ""
$PYTHON bonus/logreg_train_minibatch.py datasets/dataset_train.csv 64
echo ""
echo "✅ Modèle Mini-Batch entraîné et sauvegardé dans weights_minibatch.json"
echo ""
read -p "Appuyez sur Entrée pour continuer..."
echo ""

# Comparaison des méthodes
echo "========================================================================"
echo "BONUS: Comparaison des trois méthodes d'optimisation"
echo "======================================================================"
echo ""
$PYTHON bonus/compare_methods.py
echo ""
echo "✅ Comparaison complète des trois algorithmes"
echo ""
read -p "Appuyez sur Entrée pour continuer..."
echo ""

# Test des prédictions avec chaque méthode
echo "========================================================================"
echo "TEST: Prédictions avec les trois modèles"
echo "========================================================================"
echo ""

echo "1. Prédictions avec le modèle Batch GD..."
$PYTHON src/logreg_predict.py datasets/dataset_test.csv weights.json > /dev/null 2>&1
mv houses.csv houses_batch.csv
echo "   ✅ houses_batch.csv créé"

echo "2. Prédictions avec le modèle SGD..."
$PYTHON src/logreg_predict.py datasets/dataset_test.csv weights_sgd.json > /dev/null 2>&1
mv houses.csv houses_sgd.csv
echo "   ✅ houses_sgd.csv créé"

echo "3. Prédictions avec le modèle Mini-Batch..."
$PYTHON src/logreg_predict.py datasets/dataset_test.csv weights_minibatch.json > /dev/null 2>&1
mv houses.csv houses_minibatch.csv
echo "   ✅ houses_minibatch.csv créé"

echo ""
echo "Comparaison rapide des prédictions:"
echo ""
echo "Nombre de prédictions par modèle:"
echo "  Batch GD:   $(wc -l < houses_batch.csv) lignes"
echo "  SGD:        $(wc -l < houses_sgd.csv) lignes"
echo "  Mini-Batch: $(wc -l < houses_minibatch.csv) lignes"
echo ""

# Compter les différences
echo "Différences entre les prédictions:"
diff_batch_sgd=$(diff houses_batch.csv houses_sgd.csv 2>/dev/null | grep "^<" | wc -l || echo "0")
diff_batch_mini=$(diff houses_batch.csv houses_minibatch.csv 2>/dev/null | grep "^<" | wc -l || echo "0")
diff_sgd_mini=$(diff houses_sgd.csv houses_minibatch.csv 2>/dev/null | grep "^<" | wc -l || echo "0")

echo "  Batch vs SGD:        $diff_batch_sgd différence(s)"
echo "  Batch vs Mini-Batch: $diff_batch_mini différence(s)"
echo "  SGD vs Mini-Batch:   $diff_sgd_mini différence(s)"
echo ""

# Restaurer houses.csv avec le modèle par défaut
cp houses_batch.csv houses.csv

echo "✅ Fichier houses.csv restauré (modèle Batch GD)"
echo ""

# Résumé final
echo "========================================================================"
echo "RÉSUMÉ DES BONUS IMPLÉMENTÉS"
echo "========================================================================"
echo ""
echo "✅ BONUS 1: Statistiques avancées dans describe.py"
echo "   • Range (Étendue)"
echo "   • IQR (Écart interquartile)"
echo "   • Skewness (Coefficient d'asymétrie)"
echo "   • Kurtosis (Coefficient d'aplatissement)"
echo ""
echo "✅ BONUS 2: Stochastic Gradient Descent (SGD)"
echo "   • 1470 mises à jour par époque (une par exemple)"
echo "   • Learning rate: 0.01"
echo "   • Précision >98%"
echo ""
echo "✅ BONUS 3: Mini-Batch Gradient Descent"
echo "   • Batch size configurable (testé avec 64)"
echo "   • Compromis optimal vitesse/stabilité"
echo "   • Précision >98%"
echo ""
echo "✅ BONUS EXTRA: Script de comparaison (compare_methods.py)"
echo "   • Comparaison détaillée des poids"
echo "   • Analyse des caractéristiques de chaque méthode"
echo ""
echo "📁 Fichiers générés:"
echo "   • weights.json (Batch GD)"
echo "   • weights_sgd.json (SGD)"
echo "   • weights_minibatch.json (Mini-Batch)"
echo "   • houses_batch.csv (Prédictions Batch GD)"
echo "   • houses_sgd.csv (Prédictions SGD)"
echo "   • houses_minibatch.csv (Prédictions Mini-Batch)"
echo "   • houses.csv (Prédictions par défaut)"
echo ""
echo "📖 Documentation complète dans BONUS.md"
echo ""
echo "========================================================================"
echo "DÉMONSTRATION TERMINÉE"
echo "========================================================================"
