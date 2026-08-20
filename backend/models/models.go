package models

type LLMEvent struct {
	Model        string `json:"model"`
	Feature      string `json:"feature"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	LatencyMs    int    `json:"latency_ms"`
	PromptHash   string `json:"prompt_hash"`
}

type ModelPrice struct {
	InputPrice  float64 `json:"input_price"`
	OutputPrice float64 `json:"output_price"`
}

type EventCost struct {
	InputCost  float64 `json:"input_cost"`
	OutputCost float64 `json:"output_cost"`
	TotalCost  float64 `json:"total_cost"`
}

type AnalyticsSummary struct {
	TotalRequests int64   `json:"total_requests"`
	InputTokens   int64   `json:"input_tokens"`
	OutputTokens  int64   `json:"output_tokens"`
	TotalCost     float64 `json:"total_cost"`
}

type FeatureAnalyticsSummary struct {
	Feature   string  `json:"feature"`
	Requests  int64   `json:"requests"`
	TotalCost float64 `json:"total_cost"`
}

type ModelAnalyticsSummary struct {
	Model     string  `json:"model"`
	Requests  int64   `json:"requests"`
	TotalCost float64 `json:"total_cost"`
}

type OptimizationSummary struct {
	Type            string  `json:"type"`
	Feature         string  `json:"feature"`
	PromptHash      string  `json:"prompt_hash,omitempty"`
	AvgInputTokens  float64 `json:"avg_input_tokens"`
	AvgOutputTokens float64 `json:"avg_output_tokens,omitempty"`
	Requests        int64   `json:"requests"`
	TotalInputCost  float64 `json:"total_input_cost,omitempty"`
	TotalCost       float64 `json:"total_cost,omitempty"`
	Message         string  `json:"message"`
}
