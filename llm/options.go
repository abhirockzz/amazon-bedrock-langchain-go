package llm

import "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

type ConfigOption func(*ConfigOptions)

type ConfigOptions struct {
	DontUseHumanAssistantPrompt bool
	BedrockRuntimeClient        *bedrockruntime.Client
	ModelID                     string
}

func DontUseHumanAssistantPrompt() ConfigOption {
	return func(o *ConfigOptions) {
		o.DontUseHumanAssistantPrompt = true
	}
}

func WithBedrockRuntimeClient(client *bedrockruntime.Client) ConfigOption {
	return func(o *ConfigOptions) {
		o.BedrockRuntimeClient = client
	}
}

func WithModel(modelID string) ConfigOption {
	return func(o *ConfigOptions) {
		o.ModelID = modelID
	}
}
