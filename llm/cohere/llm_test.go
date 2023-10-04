package cohere

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tmc/langchaingo/llms"
)

func TestGenerate(t *testing.T) {

	cohereLLM, err := New("us-east-1")

	assert.Nil(t, err)

	generations, err := cohereLLM.Generate(context.Background(), []string{"what's your name?"}, llms.WithMaxTokens(100))
	assert.Nil(t, err)

	assert.Equal(t, 1, len(generations))

	assert.Contains(t, generations[0].Text, "Cohere")
}
