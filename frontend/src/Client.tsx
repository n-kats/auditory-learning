import axios from 'axios'

class Client {
  cache: Record<string, Record<number, string>>

  constructor() {
    this.cache = {}
  }

  async init(url: string) : Promise<{ request_id: string, page_num: number, valid: boolean }> {
    try {
      const response = await axios.post("/init/", { url })
        const { request_id, page_num } = response.data
        return { request_id, page_num, valid: true }
    } catch (error) {
      console.error('Error initializing document:', error)
      return { request_id: '', page_num: 0, valid: false }
    }
  }

  async explain(request_id: string, page: number): Promise<string | null> {
    try {
      const explanationResponse = await axios.post(`/explain/`, { request_id, page })
        return explanationResponse.data.explanation
    } catch (error) {
      console.error('Error fetching explanation:', error)
      return null
    }
  }

  async image(request_id: string, page: number): Promise<string | null> {
    if (this.cache[request_id] && this.cache[request_id][page]) {
      return this.cache[request_id][page]
    }
    try {
      const imageResponse = await axios.post(`/image/`, { request_id, page }, { responseType: 'blob' })
        const item = URL.createObjectURL(imageResponse.data)
          if (!this.cache[request_id]) {
            this.cache[request_id] = {}
          }
          this.cache[request_id][page] = item
            return item
    } catch (error) {
      console.error('Error fetching image:', error)
      return null
    }
  }

  async audio(request_id: string, page: number): Promise<string | null> {
    try {
      const audioResponse = await axios.post(`/audio/`, { request_id, page }, { responseType: 'blob' })
        return URL.createObjectURL(audioResponse.data)
    } catch (error) {
      console.error('Error fetching audio:', error)
      return null
    }
  }

  async regenerate(request_id: string, page: number): Promise<string|null> {
    try {
      const regenerateResponse = await axios.post(`/regenerate/`, { request_id, page })
      return regenerateResponse.data.explanation
    } catch (error) {
      console.error('Error regenerating:', error)
      return null
    }
  }

}

export default Client
