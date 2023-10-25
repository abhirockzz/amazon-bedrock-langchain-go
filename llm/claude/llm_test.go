package claude

import (
	"context"
	"testing"

	"github.com/abhirockzz/amazon-bedrock-langchain-go/llm"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/assert"
	"github.com/tmc/langchaingo/llms"
)

func TestGenerate(t *testing.T) {

	claudeLLM, err := New("us-east-1")

	assert.Nil(t, err)

	generations, err := claudeLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))

	assert.Contains(t, generations[0].Text, "Claude")
}

func TestGenerateWithUserSuppliedBedrockRuntimeClient(t *testing.T) {

	region := "us-east-1"

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	assert.Nil(t, err)

	claudeLLM, err := New(region, llm.WithBedrockRuntimeClient(bedrockruntime.NewFromConfig(cfg)))
	assert.Nil(t, err)

	generations, err := claudeLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))

	assert.Contains(t, generations[0].Text, "Claude")
}

func TestGenerateWithoutUserAssistantPromptOption(t *testing.T) {

	claudeLLM, err := New("us-east-1", llm.DontUseHumanAssistantPrompt())

	assert.Nil(t, err)

	_, err = claudeLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.NotNil(t, err)
}

func TestGenerateWithManualUserAssistantPrompt(t *testing.T) {

	claudeLLM, err := New("us-east-1", llm.DontUseHumanAssistantPrompt())

	assert.Nil(t, err)

	generations, err := claudeLLM.Generate(context.Background(), []string{"\n\nHuman:what's your name?\n\nAssistant:"}, llms.WithMaxTokens(100))

	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))

	assert.Contains(t, generations[0].Text, "Claude")
}

func TestGenerateWithoutMaxTokensOpt(t *testing.T) {

	claudeLLM, err := New("us-east-1")

	assert.Nil(t, err)

	_, err = claudeLLM.Generate(context.Background(), []string{"what's your name?"})
	assert.NotNil(t, err)
}

func TestGenerateWithUserSuppliedModelID(t *testing.T) {

	claudeLLM, err := New("us-east-1", llm.WithModel("anthropic.claude-v2"))

	assert.Nil(t, err)

	generations, err := claudeLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))

	assert.Contains(t, generations[0].Text, "Claude")
}

func TestGenerateWithUserSuppliedInvalidModelID(t *testing.T) {

	claudeLLM, err := New("us-east-1", llm.WithModel("anthropic.foobar"))

	assert.Nil(t, err)

	_, err = claudeLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.NotNil(t, err)
}
