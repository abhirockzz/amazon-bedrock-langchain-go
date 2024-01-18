package llama

import (
	"context"
	"fmt"
	"testing"

	"github.com/abhirockzz/amazon-bedrock-langchain-go/llm"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/assert"
	"github.com/tmc/langchaingo/llms"
)

func TestGenerate(t *testing.T) {

	theLLM, err := New("us-east-1")

	assert.Nil(t, err)

	generations, err := theLLM.Generate(context.Background(), []string{"[hi, what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))
	assert.Nil(t, err)

	//assert.Contains(t, generations[0].Text, "Sherry")
}

func TestGenerateWithUserSuppliedBedrockRuntimeClient(t *testing.T) {

	region := "us-east-1"

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	assert.Nil(t, err)

	theLLM, err := New(region, llm.WithBedrockRuntimeClient(bedrockruntime.NewFromConfig(cfg)))
	assert.Nil(t, err)

	generations, err := theLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))
	assert.Nil(t, err)

	//assert.Contains(t, generations[0].Text, "Sherry")
}

func TestGenerateWithUserSuppliedModelID(t *testing.T) {

	theLLM, err := New("us-east-1", llm.WithModel("meta.llama2-70b-chat-v1"))

	assert.Nil(t, err)

	generations, err := theLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))
	assert.Nil(t, err)
}

func TestGenerateWithUserSuppliedInvalidModelID(t *testing.T) {

	theLLM, err := New("us-east-1", llm.WithModel("llama.foobar"))

	assert.Nil(t, err)

	_, err = theLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.NotNil(t, err)
}

func TestGenerateWithStreamingResponse(t *testing.T) {

	claudeLLM, err := New("us-east-1")

	assert.Nil(t, err)

	generations, err := claudeLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100), llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Println(string(chunk))
		return nil
	}))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))
	assert.Nil(t, err)

	//assert.Contains(t, generations[0].Text, "Claude")
}
