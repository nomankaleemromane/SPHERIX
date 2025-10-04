export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { prompt, maxTokens = 800, model = 'gpt-4o-mini' } = req.body;

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: model,
        messages: [
          { 
            role: 'system', 
            content: 'You are an expert urban planning assistant. Reply in plain text following the headings requested by the user.' 
          },
          { 
            role: 'user', 
            content: prompt 
          }
        ],
        max_tokens: maxTokens,
        temperature: 0.7
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'OpenAI API error');
    }

    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('OpenAI API error:', error);
    res.status(500).json({ 
      error: 'Failed to get AI response',
      details: error.message 
    });
  }
}
