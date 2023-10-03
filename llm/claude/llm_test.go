package claude

import (
	"context"
	"testing"

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

func TestGenerateWithoutMaxTokensOpt(t *testing.T) {

	claudeLLM, err := New("us-east-1")

	assert.Nil(t, err)

	_, err = claudeLLM.Generate(context.Background(), []string{"what's your name?"})
	assert.NotNil(t, err)
}
