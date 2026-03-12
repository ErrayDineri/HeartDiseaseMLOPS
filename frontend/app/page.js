'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import html2canvas from 'html2canvas';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

const MODEL_KEY_TO_BACKEND = {
  svm: 'svm',
  rf: 'random_forest',
  knn: 'knn',
  logreg: 'logistic_regression',
  nn: 'neural_network'
};

const BACKEND_TO_MODEL_KEY = Object.fromEntries(
  Object.entries(MODEL_KEY_TO_BACKEND).map(([k, v]) => [v, k])
);

const ALGORITHMS = [
  { key: 'svm', label: 'SVM', doc: 'Bon pour des frontières complexes, nécessite un tuning de C et gamma.' },
  { key: 'rf', label: 'Random Forest', doc: 'Ensemble d’arbres robuste, performant avec peu de prétraitement.' },
  { key: 'knn', label: 'KNN', doc: 'Simple et efficace sur petits jeux de données bien normalisés.' },
  { key: 'logreg', label: 'Logistic Regression', doc: 'Baseline interprétable pour classification binaire.' },
  { key: 'nn', label: 'Neural Network', doc: 'Flexible mais demande plus de données et d’ajustements.' }
];

const DEFAULT_PARAMS = {
  svm: { C: 1, gamma: 0.1, kernel: 'rbf' },
  rf: { n_estimators: 200, max_depth: 8, min_samples_split: 2 },
  knn: { k: 5, weights: 'uniform' },
  logreg: { C: 1, max_iter: 300 },
  nn: { learning_rate: 0.001, epochs: 40, layers: '64-32' }
};

const EMPTY_METRICS = { accuracy: 0, f1: 0, auc: 0 };

const SAMPLE_DATA = [
  { age: 52, sex: 1, cp: 0, trestbps: 125, chol: 212, target: 0 },
  { age: 53, sex: 1, cp: 0, trestbps: 140, chol: 203, target: 0 },
  { age: 70, sex: 1, cp: 0, trestbps: 145, chol: 174, target: 0 },
  { age: 61, sex: 1, cp: 0, trestbps: 148, chol: 203, target: 0 },
  { age: 62, sex: 0, cp: 0, trestbps: 138, chol: 294, target: 0 }
];

const DEFAULT_CM = [
  { cell: 'TN', value: 0 },
  { cell: 'FP', value: 0 },
  { cell: 'FN', value: 0 },
  { cell: 'TP', value: 0 }
];

function toCsv(rows) {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  return `${cols.join(',')}\n${rows.map((r) => cols.map((c) => r[c]).join(',')).join('\n')}`;
}

function downloadFile(filename, content, type = 'text/plain') {
  const blob = new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function safeJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return '{}';
  }
}

function modelLabelFromKey(key) {
  return ALGORITHMS.find((a) => a.key === key)?.label || key;
}

function toPositiveNumber(value, fallback, min = 0.0001) {
  const num = Number(value);
  if (Number.isNaN(num) || num < min) return fallback;
  return num;
}

function toPositiveInt(value, fallback, min = 1) {
  const num = Math.round(Number(value));
  if (Number.isNaN(num) || num < min) return fallback;
  return num;
}

export default function Page() {
  const [activeTab, setActiveTab] = useState('training');
  const [selectedModels, setSelectedModels] = useState(['svm', 'rf']);
  const [trainMode, setTrainMode] = useState('scratch');
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [tuningMode, setTuningMode] = useState('grid');
  const [tuningTargetModel, setTuningTargetModel] = useState('svm');
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [lastTuningResult, setLastTuningResult] = useState(null);

  const [status, setStatus] = useState('Initialisation...');
  const [progress, setProgress] = useState(0);

  const [datasetRows, setDatasetRows] = useState(SAMPLE_DATA);
  const [selectedColumns, setSelectedColumns] = useState(Object.keys(SAMPLE_DATA[0]));
  const [selectedClasses, setSelectedClasses] = useState(['0', '1']);
  const [datasetVersion, setDatasetVersion] = useState('heart_v1');

  const [modelVersion, setModelVersion] = useState('n/a');
  const [modelsRegistry, setModelsRegistry] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState('');
  const [selectedRegistryVersion, setSelectedRegistryVersion] = useState('');
  const [mlopsAlgoFilter, setMlopsAlgoFilter] = useState('all');
  const [registryDetailMode, setRegistryDetailMode] = useState('hyperparameters');
  const [liveMetrics, setLiveMetrics] = useState({});
  const [confusionData, setConfusionData] = useState(DEFAULT_CM);
  const [predictionInputFormat, setPredictionInputFormat] = useState('json');
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionOutput, setPredictionOutput] = useState(null);

  const [tutorialOpen, setTutorialOpen] = useState(true);
  const chartsRef = useRef(null);

  const allColumns = useMemo(() => (datasetRows.length ? Object.keys(datasetRows[0]) : []), [datasetRows]);

  const filteredRows = useMemo(() => {
    let rows = [...datasetRows];
    if (allColumns.includes('target')) {
      rows = rows.filter((r) => selectedClasses.includes(String(r.target)));
    }
    return rows.map((r) => {
      const out = {};
      selectedColumns.forEach((col) => {
        out[col] = r[col];
      });
      return out;
    });
  }, [datasetRows, selectedColumns, selectedClasses, allColumns]);

  const modelCards = useMemo(
    () =>
      selectedModels.map((key) => ({
        name: modelLabelFromKey(key),
        ...(liveMetrics[key] || EMPTY_METRICS)
      })),
    [selectedModels, liveMetrics]
  );

  const selectableModelVersions = useMemo(() => {
    return modelsRegistry.filter((m) => {
      const uiKey = BACKEND_TO_MODEL_KEY[m.algorithm] || m.algorithm;
      return selectedModels.includes(uiKey);
    });
  }, [modelsRegistry, selectedModels]);

  const perfData = useMemo(
    () =>
      modelCards.map((m) => ({
        model: m.name,
        Accuracy: Number((m.accuracy * 100).toFixed(2)),
        F1: Number((m.f1 * 100).toFixed(2)),
        AUC: Number((m.auc * 100).toFixed(2))
      })),
    [modelCards]
  );

  const selectedExperiment = useMemo(() => {
    const base = mlopsAlgoFilter === 'all'
      ? experiments
      : experiments.filter((exp) => exp?.best_model?.model === mlopsAlgoFilter);
    return base.find((e) => e.id === selectedExperimentId) || base[0] || null;
  }, [experiments, selectedExperimentId, mlopsAlgoFilter]);

  const filteredModelsRegistry = useMemo(() => {
    if (mlopsAlgoFilter === 'all') return modelsRegistry;
    return modelsRegistry.filter((m) => m.algorithm === mlopsAlgoFilter);
  }, [modelsRegistry, mlopsAlgoFilter]);

  const filteredExperiments = useMemo(() => {
    if (mlopsAlgoFilter === 'all') return experiments;
    return experiments.filter((exp) => exp?.best_model?.model === mlopsAlgoFilter);
  }, [experiments, mlopsAlgoFilter]);

  const selectedRegistryModel = useMemo(
    () => filteredModelsRegistry.find((m) => m.version === selectedRegistryVersion) || filteredModelsRegistry[0] || null,
    [filteredModelsRegistry, selectedRegistryVersion]
  );

  const visualizedModel = useMemo(
    () => modelsRegistry.find((m) => m.version === selectedRegistryVersion) || null,
    [modelsRegistry, selectedRegistryVersion]
  );

  const displayedConfusionData = useMemo(() => {
    const cm = visualizedModel?.metrics?.confusion_matrix;
    if (Array.isArray(cm) && cm.length === 2 && cm[0]?.length === 2 && cm[1]?.length === 2) {
      return [
        { cell: 'TN', value: cm[0][0] },
        { cell: 'FP', value: cm[0][1] },
        { cell: 'FN', value: cm[1][0] },
        { cell: 'TP', value: cm[1][1] }
      ];
    }
    return confusionData;
  }, [visualizedModel, confusionData]);

  async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    if (!response.ok) {
      const details = await response.json().catch(() => ({}));
      throw new Error(details.detail || `API error ${response.status}`);
    }
    return response.json();
  }

  function toggleModel(key) {
    setSelectedModels((prev) => {
      if (prev.includes(key)) {
        if (prev.length === 1) return prev;
        return prev.filter((k) => k !== key);
      }
      return [...prev, key];
    });
  }

  function updateParam(modelKey, key, value) {
    setParams((prev) => ({ ...prev, [modelKey]: { ...prev[modelKey], [key]: value } }));
  }

  function applyTuningToUiParams(modelKey, bestParams) {
    const clean = Object.fromEntries(
      Object.entries(bestParams || {}).map(([k, v]) => [k.replace('estimator__', ''), v])
    );

    setParams((prev) => {
      const updated = { ...prev };

      if (modelKey === 'svm') {
        updated.svm = {
          ...updated.svm,
          ...(clean.C !== undefined ? { C: clean.C } : {}),
          ...(clean.gamma !== undefined ? { gamma: clean.gamma } : {}),
          ...(clean.kernel !== undefined ? { kernel: clean.kernel } : {})
        };
      }

      if (modelKey === 'rf') {
        updated.rf = {
          ...updated.rf,
          ...(clean.n_estimators !== undefined ? { n_estimators: clean.n_estimators } : {}),
          ...(clean.max_depth !== undefined ? { max_depth: clean.max_depth ?? '' } : {}),
          ...(clean.min_samples_split !== undefined ? { min_samples_split: clean.min_samples_split } : {})
        };
      }

      if (modelKey === 'knn') {
        updated.knn = {
          ...updated.knn,
          ...(clean.n_neighbors !== undefined ? { k: clean.n_neighbors } : {}),
          ...(clean.weights !== undefined ? { weights: clean.weights } : {})
        };
      }

      if (modelKey === 'logreg') {
        updated.logreg = {
          ...updated.logreg,
          ...(clean.C !== undefined ? { C: clean.C } : {}),
          ...(clean.max_iter !== undefined ? { max_iter: clean.max_iter } : {})
        };
      }

      if (modelKey === 'nn') {
        const layers = clean.hidden_layer_sizes;
        updated.nn = {
          ...updated.nn,
          ...(clean.learning_rate_init !== undefined ? { learning_rate: clean.learning_rate_init } : {}),
          ...(clean.max_iter !== undefined ? { epochs: Math.max(1, Math.round(Number(clean.max_iter) / 10)) } : {}),
          ...(layers !== undefined
            ? {
                layers: Array.isArray(layers)
                  ? layers.join('-')
                  : String(layers).replace(/[()\s]/g, '').replace(/,/g, '-').replace(/--+/g, '-')
              }
            : {})
        };
      }

      return updated;
    });
  }

  function uiParamsToBackend() {
    const mapped = {};
    if (selectedModels.includes('svm')) {
      const gammaRaw = params.svm.gamma;
      const gammaAsNumber = Number(gammaRaw);
      const gamma = String(gammaRaw).toLowerCase() === 'scale'
        ? 'scale'
        : Number.isNaN(gammaAsNumber) || gammaAsNumber <= 0
          ? 'scale'
          : gammaAsNumber;
      mapped.svm = {
        C: toPositiveNumber(params.svm.C, 1, 0.0001),
        gamma,
        kernel: ['rbf', 'linear'].includes(params.svm.kernel) ? params.svm.kernel : 'rbf'
      };
    }
    if (selectedModels.includes('rf')) {
      const rfDepthRaw = params.rf.max_depth;
      const rfDepthNormalized =
        rfDepthRaw === '' || String(rfDepthRaw).toLowerCase() === 'none' || rfDepthRaw == null
          ? null
          : toPositiveInt(rfDepthRaw, 8, 2);
      mapped.random_forest = {
        n_estimators: toPositiveInt(params.rf.n_estimators, 200, 10),
        max_depth: Number.isNaN(rfDepthNormalized) ? null : rfDepthNormalized,
        min_samples_split: toPositiveInt(params.rf.min_samples_split, 2, 2)
      };
    }
    if (selectedModels.includes('knn')) {
      mapped.knn = {
        n_neighbors: toPositiveInt(params.knn.k, 5, 1),
        weights: ['uniform', 'distance'].includes(params.knn.weights) ? params.knn.weights : 'uniform'
      };
    }
    if (selectedModels.includes('logreg')) {
      mapped.logistic_regression = {
        C: toPositiveNumber(params.logreg.C, 1.0, 0.0001),
        max_iter: toPositiveInt(params.logreg.max_iter, 300, 50)
      };
    }
    if (selectedModels.includes('nn')) {
      const layers = String(params.nn.layers)
        .split('-')
        .map((v) => Number(v.trim()))
        .filter((v) => !Number.isNaN(v) && v > 0)
        .map((v) => Math.round(v));
      mapped.neural_network = {
        hidden_layer_sizes: layers.length ? layers : [64, 32],
        learning_rate_init: toPositiveNumber(params.nn.learning_rate, 0.001, 0.000001),
        max_iter: Math.max(100, toPositiveInt(params.nn.epochs, 40, 1) * 10)
      };
    }
    return mapped;
  }

  async function refreshExperimentsAndModel() {
    const [expRes, modelsRes] = await Promise.all([api('/experiments'), api('/models')]);
    setExperiments(expRes.experiments || []);
    setModelsRegistry(modelsRes.models || []);
    const active = (modelsRes.models || []).find((m) => m.is_active);
    if (active) {
      setModelVersion(active.version);
      setSelectedRegistryVersion(active.version);
    }
  }

  async function refreshDatasetPreview(version, limit = 120) {
    const preview = await api(`/datasets/${version}/preview?limit=${limit}`);
    const rows = preview.rows || [];
    if (rows.length) {
      setDatasetRows(rows);
      setSelectedColumns(preview.columns || Object.keys(rows[0]));
      const firstRow = rows[0] || {};
      const featuresOnly = Object.fromEntries(Object.entries(firstRow).filter(([k]) => k !== 'target'));
      setPredictionInput(JSON.stringify([featuresOnly], null, 2));
      setPredictionOutput(null);
    }
  }

  function applyResult(result) {
    const metricsMap = {};
    (result.results || []).forEach((row) => {
      const key = BACKEND_TO_MODEL_KEY[row.model] || row.model;
      metricsMap[key] = {
        accuracy: row.metrics.accuracy,
        f1: row.metrics.f1,
        auc: row.metrics.roc_auc || 0
      };
    });
    setLiveMetrics((prev) => ({ ...prev, ...metricsMap }));

    const best = result.best_model;
    if (best?.metrics?.confusion_matrix) {
      const cm = best.metrics.confusion_matrix;
      setConfusionData([
        { cell: 'TN', value: cm[0][0] },
        { cell: 'FP', value: cm[0][1] },
        { cell: 'FN', value: cm[1][0] },
        { cell: 'TP', value: cm[1][1] }
      ]);
    }
    if (best?.model_version) {
      setModelVersion(best.model_version);
      setSelectedRegistryVersion(best.model_version);
    }
    if (result?.experiment_id) setSelectedExperimentId(result.experiment_id);
  }

  function clearModelOutputs() {
    setLiveMetrics({});
    setConfusionData(DEFAULT_CM);
    setModelVersion('n/a');
    setSelectedExperimentId('');
    setSelectedRegistryVersion('');
    setPredictionOutput(null);
  }

  function parsePredictionRecords() {
    const raw = predictionInput.trim();
    if (!raw) {
      throw new Error('Entrée vide. Fournis au moins un enregistrement JSON ou CSV.');
    }

    if (predictionInputFormat === 'json') {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        if (!parsed.length) throw new Error('Le tableau JSON est vide.');
        return parsed;
      }
      if (parsed && typeof parsed === 'object') {
        return [parsed];
      }
      throw new Error('Format JSON invalide: utilise un objet ou un tableau d’objets.');
    }

    const lines = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length < 2) {
      throw new Error('Format CSV invalide: ajoute un header et au moins une ligne de valeurs.');
    }

    const headers = lines[0].split(',').map((h) => h.trim());
    const rows = lines.slice(1).map((line) => {
      const values = line.split(',').map((v) => v.trim());
      const record = {};
      headers.forEach((header, index) => {
        const value = values[index] ?? '';
        const asNumber = Number(value);
        record[header] = value !== '' && !Number.isNaN(asNumber) ? asNumber : value;
      });
      return record;
    });

    return rows;
  }

  async function runPredictionWithSelectedVersion() {
    const version = selectedRegistryVersion || (modelVersion !== 'n/a' ? modelVersion : '');
    if (!version) {
      setStatus('Aucune version modèle sélectionnée. Charge un modèle pré-entraîné ou entraîne d’abord.');
      return;
    }

    try {
      const records = parsePredictionRecords();
      setStatus(`Prédiction en cours sur ${version}...`);
      setProgress(70);

      const result = await api(`/models/${version}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records })
      });

      setPredictionOutput({
        version,
        records_count: records.length,
        predictions: result.predictions,
        probabilities: result.probabilities
      });
      setProgress(100);
      setStatus(`Prédiction terminée sur ${version} (${records.length} ligne(s)).`);
    } catch (error) {
      setProgress(0);
      setStatus(`Prédiction impossible: ${error.message}`);
    }
  }

  async function loadSelectedPretrainedModel() {
    if (!selectedRegistryModel?.version) {
      setStatus('Aucune version disponible. Entraîne un modèle ou choisis une version existante.');
      return;
    }

    try {
      setStatus(`Chargement du modèle pré-entraîné ${selectedRegistryModel.version}...`);
      setProgress(40);

      await api(`/models/${selectedRegistryModel.version}/rollback`, { method: 'POST' });
      await refreshExperimentsAndModel();

      const uiKey = BACKEND_TO_MODEL_KEY[selectedRegistryModel.algorithm] || selectedRegistryModel.algorithm;
      setSelectedModels((prev) => (prev.includes(uiKey) ? prev : [...prev, uiKey]));
      setLiveMetrics((prev) => ({
        ...prev,
        [uiKey]: {
          accuracy: selectedRegistryModel.metrics?.accuracy ?? 0,
          f1: selectedRegistryModel.metrics?.f1 ?? 0,
          auc: selectedRegistryModel.metrics?.roc_auc ?? 0
        }
      }));
      setModelVersion(selectedRegistryModel.version);
      setProgress(100);
      setStatus(`Modèle pré-entraîné chargé: ${selectedRegistryModel.version} (${selectedRegistryModel.algorithm}).`);
    } catch (error) {
      setProgress(0);
      setStatus(`Chargement pré-entraîné impossible: ${error.message}`);
    }
  }

  async function runTraining() {
    if (trainMode === 'pretrained') {
      await loadSelectedPretrainedModel();
      return;
    }

    try {
      setStatus('Entraînement en cours...');
      setProgress(20);
      const payload = {
        dataset_version: datasetVersion,
        models: selectedModels.map((m) => MODEL_KEY_TO_BACKEND[m]),
        target_column: 'target',
        selected_columns: selectedColumns.filter((c) => c !== 'target'),
        selected_classes: selectedClasses.map(Number),
        drop_duplicates: true,
        clean_missing: 'none',
        hyperparameters: uiParamsToBackend(),
        optimize_metric: 'f1'
      };
      const result = await api('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      applyResult(result);
      await refreshExperimentsAndModel();
      setProgress(100);
      setStatus(`Training terminé. Meilleur modèle: ${result.best_model.model}`);
    } catch (error) {
      setProgress(0);
      setStatus(`Erreur entraînement: ${error.message}`);
    }
  }

  async function runAutoTuning(mode) {
    try {
      setTuningMode(mode);
      setStatus(`Tuning ${mode} en cours...`);
      setProgress(35);
      const model = selectedModels.includes(tuningTargetModel) ? tuningTargetModel : selectedModels[0] || 'svm';
      const result = await api('/tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_version: datasetVersion,
          model: MODEL_KEY_TO_BACKEND[model],
          method: mode,
          target_column: 'target',
          n_iter: 15,
          n_trials: 15,
          cv: 3,
          scoring: 'f1',
          selected_columns: selectedColumns.filter((c) => c !== 'target'),
          selected_classes: selectedClasses.map(Number),
          clean_missing: 'none',
          drop_duplicates: true
        })
      });

      applyTuningToUiParams(model, result.best_params);
      setLastTuningResult({ ...result, model_ui_key: model });
      setProgress(100);
      setStatus(`Tuning ${mode} terminé - score CV ${result.best_cv_score.toFixed(4)} (config mise à jour)`);
    } catch (error) {
      setProgress(0);
      setStatus(`Erreur tuning: ${error.message}`);
    }
  }

  async function runTrainingWithTunedConfig() {
    if (!lastTuningResult) {
      setStatus('Aucun tuning récent trouvé. Lance un tuning d’abord.');
      return;
    }
    await runTraining();
  }

  async function runAutoML() {
    try {
      setStatus('AutoML en cours...');
      setProgress(45);
      const result = await api('/automl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_version: datasetVersion,
          target_column: 'target',
          candidate_models: selectedModels.map((m) => MODEL_KEY_TO_BACKEND[m]),
          optimize_metric: 'f1',
          quick_tuning: true,
          test_size: 0.2,
          selected_columns: selectedColumns.filter((c) => c !== 'target'),
          selected_classes: selectedClasses.map(Number),
          clean_missing: 'none',
          drop_duplicates: true
        })
      });
      applyResult({ best_model: result.best_model, results: [result.best_model] });
      const bestKey = BACKEND_TO_MODEL_KEY[result.best_model.model];
      if (bestKey) setSelectedModels([bestKey]);
      await refreshExperimentsAndModel();
      setProgress(100);
      setStatus(`AutoML terminé: meilleur modèle ${result.best_model.model}`);
    } catch (error) {
      setProgress(0);
      setStatus(`Erreur AutoML: ${error.message}`);
    }
  }

  function saveCurrentConfig() {
    const entry = {
      id: `cfg_${savedConfigs.length + 1}`,
      models: selectedModels.join(','),
      mode: trainMode,
      tuning: tuningMode,
      params: JSON.stringify(params)
    };
    setSavedConfigs((prev) => [entry, ...prev]);
    setStatus(`Configuration ${entry.id} sauvegardée`);
  }

  function loadConfig(cfg) {
    setSelectedModels(cfg.models.split(',').filter(Boolean));
    setTrainMode(cfg.mode);
    setTuningMode(cfg.tuning);
    setParams(JSON.parse(cfg.params));
    setStatus(`Configuration ${cfg.id} chargée`);
  }

  async function rollbackVersion(version) {
    if (!version) return;
    try {
      await api(`/models/${version}/rollback`, { method: 'POST' });
      setModelVersion(version);
      setStatus(`Rollback vers ${version} effectué`);
      await refreshExperimentsAndModel();
    } catch (error) {
      setStatus(`Rollback impossible: ${error.message}`);
    }
  }

  async function cleanMissingValues() {
    try {
      const result = await api('/datasets/clean', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version: datasetVersion, strategy: 'dropna' })
      });
      const newVersion = result.dataset.version;
      clearModelOutputs();
      setDatasetVersion(newVersion);
      await refreshDatasetPreview(newVersion);
      setStatus(`Nettoyage OK. Dataset actif: ${newVersion}. Les métriques affichées ont été réinitialisées.`);
    } catch (error) {
      setStatus(`Nettoyage impossible: ${error.message}`);
    }
  }

  async function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${API_BASE}/datasets/upload`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Upload échoué');
      const payload = await response.json();
      const version = payload.dataset.version;
      clearModelOutputs();
      setDatasetVersion(version);
      await refreshDatasetPreview(version);
      setStatus(`Upload OK. Dataset actif: ${version}. Les métriques affichées ont été réinitialisées.`);
    } catch (error) {
      setStatus(`Upload impossible: ${error.message}`);
    }
  }

  async function exportPng() {
    if (!chartsRef.current) return;
    const canvas = await html2canvas(chartsRef.current);
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = 'dashboard_charts.png';
    a.click();
    setStatus('Export PNG effectué');
  }

  function exportResultsCsv() {
    downloadFile('model_results.csv', toCsv(perfData), 'text/csv;charset=utf-8;');
    setStatus('Export CSV des résultats effectué');
  }

  function exportModel() {
    if (!modelVersion || modelVersion === 'n/a') {
      setStatus('Aucun modèle exportable. Lance un entraînement.');
      return;
    }
    window.open(`${API_BASE}/models/${modelVersion}/download`, '_blank');
    setStatus(`Téléchargement du modèle ${modelVersion} lancé`);
  }

  async function resetAllStorage() {
    const ok = window.confirm(
      'Confirmer la suppression de toutes les sauvegardes (modèles, expériences, datasets versionnés) ?'
    );
    if (!ok) return;

    try {
      setStatus('Reset global en cours...');
      setProgress(20);
      await api('/admin/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keep_default_dataset: true })
      });

      setSavedConfigs([]);
      setLastTuningResult(null);
      setLiveMetrics({});
      setConfusionData(DEFAULT_CM);
      setPredictionOutput(null);
      setModelVersion('n/a');
      setSelectedExperimentId('');
      setSelectedRegistryVersion('');

      const ds = await api('/datasets');
      const active = ds.active || 'heart_v1';
      setDatasetVersion(active);
      await refreshDatasetPreview(active);
      await refreshExperimentsAndModel();

      setProgress(100);
      setStatus('Reset terminé: sauvegardes, modèles et expériences supprimés. Dataset par défaut restauré.');
      setActiveTab('mlops');
    } catch (error) {
      setProgress(0);
      setStatus(`Reset impossible: ${error.message}`);
    }
  }

  useEffect(() => {
    async function bootstrap() {
      try {
        const ds = await api('/datasets');
        const active = ds.active || 'heart_v1';
        setDatasetVersion(active);
        await refreshDatasetPreview(active);
        await refreshExperimentsAndModel();
        setStatus('Connecté au backend FastAPI');
      } catch {
        setStatus(`Backend non joignable (${API_BASE}). Démarre FastAPI pour activer le training réel.`);
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    if (!selectedModels.length) {
      setTuningTargetModel('svm');
      return;
    }
    if (!selectedModels.includes(tuningTargetModel)) {
      setTuningTargetModel(selectedModels[0]);
    }
  }, [selectedModels, tuningTargetModel]);

  useEffect(() => {
    if (filteredExperiments.length && !selectedExperimentId) {
      setSelectedExperimentId(filteredExperiments[0].id);
      return;
    }
    if (selectedExperimentId && !filteredExperiments.some((exp) => exp.id === selectedExperimentId)) {
      setSelectedExperimentId(filteredExperiments[0]?.id || '');
    }
  }, [filteredExperiments, selectedExperimentId]);

  useEffect(() => {
    if (filteredModelsRegistry.length && !selectedRegistryVersion) {
      setSelectedRegistryVersion(filteredModelsRegistry[0].version);
      return;
    }
    if (selectedRegistryVersion && !filteredModelsRegistry.some((m) => m.version === selectedRegistryVersion)) {
      setSelectedRegistryVersion(filteredModelsRegistry[0]?.version || '');
    }
  }, [filteredModelsRegistry, selectedRegistryVersion]);

  useEffect(() => {
    if (!selectableModelVersions.length) return;
    const stillValid = selectableModelVersions.some((m) => m.version === selectedRegistryVersion);
    if (!selectedRegistryVersion || !stillValid) {
      setSelectedRegistryVersion(selectableModelVersions[0].version);
    }
  }, [selectableModelVersions, selectedRegistryVersion]);

  return (
    <main className="page">
      <header className="header card">
        <div>
          <h1 className="title">Heart Disease ML Studio</h1>
          <div className="subtle">Interface allégée en onglets: Training, Data, MLOps</div>
        </div>
        <div className="row">
          <button onClick={runTraining}>Lancer entraînement</button>
          <button className="primary" onClick={runAutoML}>Mode AutoML</button>
          <button onClick={exportModel}>Exporter modèle</button>
        </div>
      </header>

      <div className="toast">Statut: {status} | Progression training: {progress}%</div>

      <section className="card tabs">
        <button className={activeTab === 'training' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('training')}>Training</button>
        <button className={activeTab === 'data' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('data')}>Data</button>
        <button className={activeTab === 'mlops' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('mlops')}>MLOps</button>
      </section>

      {tutorialOpen && (
        <section className="card">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong>Tutoriel rapide</strong>
            <button onClick={() => setTutorialOpen(false)}>Fermer</button>
          </div>
          <div className="tooltip">
            {activeTab === 'training' && 'Sélectionne des modèles, tune, puis entraîne. Les résultats de tuning mettent à jour la configuration.'}
            {activeTab === 'data' && 'Upload, filtre colonnes/classes et nettoie le dataset avant entraînement.'}
            {activeTab === 'mlops' && 'Visualise versions, hyperparamètres, métriques, expérimentations et rollback.'}
          </div>
        </section>
      )}

      {activeTab === 'training' && (
        <section className="grid" ref={chartsRef}>
          <div className="card span-12">
            <h2>Workflow guidé</h2>
            <div className="step-list">
              <div className="step-item"><strong>1.</strong> Choisis les modèles à comparer.</div>
              <div className="step-item"><strong>2.</strong> Cible un modèle pour le tuning.</div>
              <div className="step-item"><strong>3.</strong> Vérifie les hyperparamètres auto-appliqués.</div>
              <div className="step-item"><strong>4.</strong> Lance l’entraînement et consulte MLOps.</div>
            </div>
          </div>

          <div className="card span-5">
            <h2>Sélection et gestion des modèles</h2>
            <div className="row" style={{ marginBottom: 10 }}>
              {ALGORITHMS.map((algo) => (
                <button
                  key={algo.key}
                  className={selectedModels.includes(algo.key) ? 'chip active' : 'chip'}
                  onClick={() => toggleModel(algo.key)}
                >
                  {algo.label}
                </button>
              ))}
            </div>
            <label>Mode d’entraînement</label>
            <select value={trainMode} onChange={(e) => setTrainMode(e.target.value)}>
              <option value="scratch">Entraîner depuis zéro</option>
              <option value="pretrained">Charger modèle pré-entraîné</option>
            </select>

            <div className="row" style={{ marginTop: 10 }}>
              <div style={{ flex: 1 }}>
                <label>Version modèle (versionning)</label>
                <select value={selectedRegistryVersion || ''} onChange={(e) => setSelectedRegistryVersion(e.target.value)}>
                  {selectableModelVersions.map((m) => (
                    <option key={m.version} value={m.version}>{m.version} | {m.algorithm} | ds:{m.dataset_version}</option>
                  ))}
                  {!selectableModelVersions.length && <option value="">Aucune version compatible</option>}
                </select>
              </div>
            </div>

            <div className="row" style={{ marginTop: 8 }}>
              <button onClick={loadSelectedPretrainedModel} disabled={!selectedRegistryVersion}>Charger la version sélectionnée</button>
            </div>

            {trainMode === 'pretrained' && (
              <div className="tooltip" style={{ marginTop: 8 }}>
                Réutiliser un modèle pré-entraîné = activer une version déjà sauvegardée (rollback), réafficher ses métriques finales,
                puis l’utiliser directement pour prédiction/export sans relancer un nouvel entraînement.
              </div>
            )}
            <div className="column" style={{ marginTop: 10 }}>
              {ALGORITHMS.map((algo) => (
                <div key={algo.key} className="tooltip">• <strong>{algo.label}:</strong> {algo.doc}</div>
              ))}
            </div>
          </div>

          <div className="card span-7">
            <h2>Paramétrage avancé</h2>
            <div className="grid">
              {selectedModels.map((modelKey) => (
                <div className="span-6" key={modelKey}>
                  <label><strong>{modelLabelFromKey(modelKey)}</strong></label>
                  <div className="column">
                    {Object.entries(params[modelKey] || {}).map(([k, v]) => (
                      <div key={k}>
                        <label>{k}</label>
                        <input value={v} onChange={(e) => updateParam(modelKey, k, e.target.value)} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <button onClick={() => runAutoTuning('grid')}>Tuning GridSearch</button>
              <button onClick={() => runAutoTuning('random')}>Tuning RandomSearch</button>
              <button onClick={() => runAutoTuning('optuna')}>Tuning Optuna</button>
              <button onClick={() => runAutoTuning('automl_quick')}>Tuning AutoML-like</button>
              <button className="primary" onClick={saveCurrentConfig}>Sauvegarder config</button>
            </div>

            {lastTuningResult && (
              <div className="json-box" style={{ marginTop: 10 }}>
                <strong>Dernier résultat de tuning ({lastTuningResult.method})</strong>
                <div className="subtle">Modèle: {modelLabelFromKey(lastTuningResult.model_ui_key)} | Best CV: {lastTuningResult.best_cv_score?.toFixed?.(4)}</div>
                <pre>{safeJson(lastTuningResult.best_params)}</pre>
              </div>
            )}

            <div className="row" style={{ marginTop: 10 }}>
              <label style={{ minWidth: 170 }}>Modèle ciblé pour tuning</label>
              <select value={tuningTargetModel} onChange={(e) => setTuningTargetModel(e.target.value)} style={{ maxWidth: 260 }}>
                {selectedModels.map((key) => (
                  <option key={key} value={key}>{modelLabelFromKey(key)}</option>
                ))}
              </select>
            </div>

            <div className="row" style={{ marginTop: 8 }}>
              <button onClick={runTrainingWithTunedConfig}>Entraîner avec config tunée</button>
            </div>

            <div className="column" style={{ marginTop: 8 }}>
              {savedConfigs.map((cfg) => (
                <div className="row" key={cfg.id}>
                  <span className="chip">{cfg.id} | {cfg.models} | {cfg.tuning}</span>
                  <button onClick={() => loadConfig(cfg)}>Charger</button>
                </div>
              ))}
            </div>
          </div>

          <div className="card span-12">
            <h2>Visualisation et analyse</h2>
            <div className="row" style={{ marginBottom: 10, gap: 12 }}>
              <label style={{ minWidth: 180 }}>Modèle sélectionné</label>
              {visualizedModel ? (
                <span>
                  <strong>{visualizedModel.algorithm}</strong>{' '}
                  <span className="subtle">version: {visualizedModel.version} | dataset: {visualizedModel.dataset_version}</span>
                </span>
              ) : (
                <span className="subtle">Aucune version sélectionnée</span>
              )}
            </div>
            {visualizedModel?.metrics ? (
              <div className="kpis" style={{ marginBottom: 10, gridTemplateColumns: 'repeat(9, minmax(120px, 1fr))' }}>
                <div className="kpi"><div className="subtle">Accuracy</div><strong>{((visualizedModel.metrics.accuracy ?? 0) * 100).toFixed(1)}%</strong></div>
                <div className="kpi"><div className="subtle">Precision</div><strong>{((visualizedModel.metrics.precision ?? 0) * 100).toFixed(1)}%</strong></div>
                <div className="kpi"><div className="subtle">Recall</div><strong>{((visualizedModel.metrics.recall ?? 0) * 100).toFixed(1)}%</strong></div>
                <div className="kpi"><div className="subtle">F1</div><strong>{((visualizedModel.metrics.f1 ?? 0) * 100).toFixed(1)}%</strong></div>
                <div className="kpi"><div className="subtle">ROC AUC</div><strong>{((visualizedModel.metrics.roc_auc ?? 0) * 100).toFixed(1)}%</strong></div>
                <div className="kpi"><div className="subtle">Error Rate</div><strong>{((visualizedModel.metrics.error_rate ?? 0) * 100).toFixed(1)}%</strong></div>
                <div className="kpi"><div className="subtle">MAE</div><strong>{visualizedModel.metrics.mae != null ? Number(visualizedModel.metrics.mae).toFixed(3) : '-'}</strong></div>
                <div className="kpi"><div className="subtle">MSE</div><strong>{visualizedModel.metrics.mse != null ? Number(visualizedModel.metrics.mse).toFixed(3) : '-'}</strong></div>
                <div className="kpi"><div className="subtle">RMSE</div><strong>{visualizedModel.metrics.rmse != null ? Number(visualizedModel.metrics.rmse).toFixed(3) : '-'}</strong></div>
              </div>
            ) : (
              <div className="subtle" style={{ marginBottom: 10 }}>Aucun modèle sélectionné. Entraîne ou sélectionne une version.</div>
            )}
            <div className="grid">
              <div className="span-6" style={{ minHeight: 270 }}>
                <label>Matrice de confusion</label>
                <div className="subtle" style={{ marginBottom: 6 }}>
                  Basée sur la version de modèle sélectionnée ci-dessus.
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={displayedConfusionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="cell" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="span-6" style={{ minHeight: 270 }}>
                <label>Performance multi-modèles</label>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={perfData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Accuracy" fill="#2563eb" />
                    <Bar dataKey="F1" fill="#7c3aed" />
                    <Bar dataKey="AUC" fill="#0d9488" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="span-12" style={{ minHeight: 280 }}>
                <label>Comparaison radar</label>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart
                    data={[
                      { metric: 'Accuracy', ...Object.fromEntries(modelCards.map((m) => [m.name, m.accuracy * 100])) },
                      { metric: 'F1', ...Object.fromEntries(modelCards.map((m) => [m.name, m.f1 * 100])) },
                      { metric: 'AUC', ...Object.fromEntries(modelCards.map((m) => [m.name, m.auc * 100])) }
                    ]}
                  >
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis domain={[0, 100]} />
                    {modelCards.map((m, idx) => (
                      <Radar
                        key={m.name}
                        name={m.name}
                        dataKey={m.name}
                        stroke={['#2563eb', '#7c3aed', '#0d9488', '#f59e0b'][idx % 4]}
                        fill={['#2563eb', '#7c3aed', '#0d9488', '#f59e0b'][idx % 4]}
                        fillOpacity={0.2}
                      />
                    ))}
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <button onClick={exportPng}>Exporter graphiques PNG</button>
              <button onClick={exportResultsCsv}>Exporter résultats CSV</button>
            </div>
          </div>

          <div className="card span-12">
            <h2>Prédiction avec modèle pré-entraîné (versionning)</h2>
            <div className="row" style={{ marginBottom: 8 }}>
              <div className="chip">Version ciblée: {selectedRegistryVersion || modelVersion || 'n/a'}</div>
              <div className="subtle">Utilise la version chargée pour prédire sans ré-entraîner.</div>
            </div>

            <div className="row" style={{ marginBottom: 8 }}>
              <label style={{ minWidth: 70 }}>Format</label>
              <select value={predictionInputFormat} onChange={(e) => setPredictionInputFormat(e.target.value)} style={{ maxWidth: 180 }}>
                <option value="json">JSON</option>
                <option value="csv">CSV</option>
              </select>
              <button onClick={runPredictionWithSelectedVersion}>Lancer prédiction</button>
            </div>

            <textarea
              value={predictionInput}
              onChange={(e) => setPredictionInput(e.target.value)}
              style={{ minHeight: 150, width: '100%' }}
              placeholder={
                predictionInputFormat === 'json'
                  ? '[{"age":52,"sex":1,"cp":0,...}] ou {"age":52,...}'
                  : 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n52,1,0,125,212,0,1,168,0,1.0,2,2,3'
              }
            />

            <div className="tooltip" style={{ marginTop: 8 }}>
              Réutiliser un modèle pré-entraîné = charger une version existante, la rendre active, puis appeler directement l’endpoint de prédiction avec tes nouvelles observations.
            </div>

            {predictionOutput && (
              <pre className="json-box" style={{ marginTop: 8 }}>{safeJson(predictionOutput)}</pre>
            )}
          </div>
        </section>
      )}

      {activeTab === 'data' && (
        <section className="grid">
          <div className="card span-12">
            <h2>Interaction avec les données</h2>
            <div className="row">
              <div style={{ flex: 1 }}>
                <label>Upload nouvelles données (CSV)</label>
                <input type="file" accept=".csv" onChange={onUpload} />
              </div>
              <button onClick={cleanMissingValues}>Nettoyage rapide (drop NaN)</button>
            </div>

            <div className="split-grid" style={{ marginTop: 10 }}>
              <div>
                <label>Colonnes sélectionnées</label>
                <select
                  multiple
                  value={selectedColumns}
                  onChange={(e) => setSelectedColumns(Array.from(e.target.selectedOptions).map((o) => o.value))}
                  style={{ minHeight: 120 }}
                >
                  {allColumns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              {allColumns.includes('target') && (
                <div>
                  <label>Classes target</label>
                  <select
                    multiple
                    value={selectedClasses}
                    onChange={(e) => setSelectedClasses(Array.from(e.target.selectedOptions).map((o) => o.value))}
                    style={{ minHeight: 120 }}
                  >
                    <option value="0">0</option>
                    <option value="1">1</option>
                  </select>
                </div>
              )}
            </div>

            <div className="table-wrap" style={{ marginTop: 10 }}>
              <table>
                <thead>
                  <tr>
                    {selectedColumns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.slice(0, 12).map((row, idx) => (
                    <tr key={idx}>
                      {selectedColumns.map((c) => (
                        <td key={c}>{String(row[c] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="subtle" style={{ marginTop: 6 }}>
              Dataset version: {datasetVersion} | Prévisualisation: {Math.min(filteredRows.length, 12)} / {filteredRows.length}
            </div>
          </div>
        </section>
      )}

      {activeTab === 'mlops' && (
        <section className="grid">
          <div className="card span-12">
            <h2>Vue MLOps complète</h2>
            <div className="kpis" style={{ marginBottom: 12 }}>
              <div className="kpi"><div className="subtle">Model version active</div><strong>{modelVersion}</strong></div>
              <div className="kpi"><div className="subtle">Dataset version active</div><strong>{datasetVersion}</strong></div>
              <div className="kpi"><div className="subtle">Dernier tuning</div><strong>{tuningMode}</strong></div>
              <div className="kpi"><div className="subtle">Expériences suivies</div><strong>{filteredExperiments.length}</strong></div>
            </div>

            <div className="row" style={{ marginBottom: 12 }}>
              <button className="danger" onClick={resetAllStorage}>Tout supprimer (sauvegardes, modèles, expériences)</button>
            </div>

            <div className="row" style={{ marginBottom: 12 }}>
              <label style={{ minWidth: 140 }}>Filtre algorithme</label>
              <select value={mlopsAlgoFilter} onChange={(e) => setMlopsAlgoFilter(e.target.value)} style={{ maxWidth: 280 }}>
                <option value="all">Tous</option>
                {Object.values(MODEL_KEY_TO_BACKEND).map((algo) => (
                  <option key={algo} value={algo}>{algo}</option>
                ))}
              </select>
            </div>

            {lastTuningResult && (
              <div className="json-box" style={{ marginBottom: 12 }}>
                <strong>Résultat de tuning le plus récent</strong>
                <div className="subtle">
                  Modèle: {modelLabelFromKey(lastTuningResult.model_ui_key)} | méthode: {lastTuningResult.method} | best_cv_score: {lastTuningResult.best_cv_score?.toFixed?.(4)}
                </div>
                <pre>{safeJson(lastTuningResult.best_params)}</pre>
              </div>
            )}

            <div className="grid">
              <div className="card span-6">
                <h2>Registry des modèles</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Version</th>
                        <th>Algo</th>
                        <th>Dataset</th>
                        <th>F1</th>
                        <th>Actif</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredModelsRegistry.map((m) => (
                        <tr key={m.version}>
                          <td>{m.version}</td>
                          <td>{m.algorithm}</td>
                          <td>{m.dataset_version}</td>
                          <td>{m.metrics?.f1?.toFixed ? m.metrics.f1.toFixed(3) : '-'}</td>
                          <td>{m.is_active ? 'Oui' : 'Non'}</td>
                          <td>
                            <div className="row">
                              <button
                                onClick={() => {
                                  setSelectedRegistryVersion(m.version);
                                  setRegistryDetailMode('hyperparameters');
                                }}
                              >
                                Hyperparamètres
                              </button>
                              <button
                                onClick={() => {
                                  setSelectedRegistryVersion(m.version);
                                  setRegistryDetailMode('details');
                                }}
                              >
                                Détails
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {selectedRegistryModel && (
                  <pre className="json-box" style={{ marginTop: 8 }}>{safeJson(
                    registryDetailMode === 'hyperparameters'
                      ? {
                          version: selectedRegistryModel.version,
                          algorithm: selectedRegistryModel.algorithm,
                          hyperparameters: selectedRegistryModel.hyperparameters
                        }
                      : {
                          version: selectedRegistryModel.version,
                          algorithm: selectedRegistryModel.algorithm,
                          dataset_version: selectedRegistryModel.dataset_version,
                          metrics: selectedRegistryModel.metrics,
                          is_active: selectedRegistryModel.is_active,
                          created_at: selectedRegistryModel.created_at,
                          experiment_id: selectedRegistryModel.experiment_id
                        }
                  )}</pre>
                )}
              </div>

              <div className="card span-6">
                <h2>Historique des expérimentations</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Best model</th>
                        <th>F1</th>
                        <th>Version</th>
                        <th>Détails</th>
                        <th>Rollback</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredExperiments.map((exp, idx) => (
                        <tr key={`${exp.id}_${idx}`}>
                          <td>{exp.id}</td>
                          <td>{exp.best_model?.model || '-'}</td>
                          <td>{exp.best_model?.metrics?.f1?.toFixed ? exp.best_model.metrics.f1.toFixed(3) : '-'}</td>
                          <td>{exp.best_model?.model_version || '-'}</td>
                          <td>
                            <button onClick={() => setSelectedExperimentId(exp.id)}>Détails</button>
                          </td>
                          <td>
                            <button
                              className="warn"
                              onClick={() => rollbackVersion(exp.best_model?.model_version)}
                              disabled={!exp.best_model?.model_version}
                            >
                              Rollback
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {selectedExperiment && (
                  <pre className="json-box" style={{ marginTop: 8 }}>{safeJson(selectedExperiment)}</pre>
                )}
              </div>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
