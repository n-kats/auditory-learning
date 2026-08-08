import { Button, Group, TextInput } from '@mantine/core';
import { useForm } from '@mantine/form';
import Client from './Client.tsx'

interface UrlInputGetResultProps {
  request_id: string;
  page_num: number;
}


function UrlInput({ onGetResult }: { onGetResult: (result: UrlInputGetResultProps) => void }) {
  const form = useForm({
    mode: 'uncontrolled',
    initialValues: {
      url: '',
    },

    validate: {
      url: (value) => {
        if (!value) {
          return 'URL is required';
        }

        if (!value.startsWith('http')) {
          return 'URL must start with http';
        }
      },
    }
  })

  const submit = form.onSubmit((values) => {
    const client = new Client()
    client.init(values.url).then(({ request_id, page_num, valid }) => {
      if (!valid) {
        return
      }
      onGetResult({ request_id, page_num})
    })
  })

  return (
    <form onSubmit={submit}>
      <TextInput
        label="URL"
        placeholder="https://arxiv.org/pdf/..."
        key={form.key('url')}
        {...form.getInputProps('url')}
      />
      <Group justify="flex-end" mt="md">
        <Button type="submit">開始</Button>
      </Group>
    </form>

  )
}

export default UrlInput
