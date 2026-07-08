{{/*
Expand the name of the chart.
*/}}
{{- define "bedrock-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec).
*/}}
{{- define "bedrock-gateway.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "bedrock-gateway.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "bedrock-gateway.labels" -}}
helm.sh/chart: {{ include "bedrock-gateway.chart" . }}
{{ include "bedrock-gateway.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "bedrock-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "bedrock-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "bedrock-gateway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "bedrock-gateway.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
The resolved image tag: image.tag, falling back to the chart appVersion.
*/}}
{{- define "bedrock-gateway.imageTag" -}}
{{- default .Chart.AppVersion .Values.image.tag }}
{{- end }}

{{/*
The name of the API_KEY Secret the Deployment references. When apiKey.value is
set, the chart-managed Secret; when apiKey.existingSecret is set, that secret.
*/}}
{{- define "bedrock-gateway.apiKeySecretName" -}}
{{- if .Values.apiKey.existingSecret }}
{{- .Values.apiKey.existingSecret }}
{{- else }}
{{- printf "%s-apikey" (include "bedrock-gateway.fullname" .) }}
{{- end }}
{{- end }}

{{/*
The key within the API_KEY Secret. For an existing secret, honor
existingSecretKey (default API_KEY); the chart-managed Secret always uses
API_KEY.
*/}}
{{- define "bedrock-gateway.apiKeySecretKey" -}}
{{- if .Values.apiKey.existingSecret }}
{{- default "API_KEY" .Values.apiKey.existingSecretKey }}
{{- else }}
{{- "API_KEY" }}
{{- end }}
{{- end }}
