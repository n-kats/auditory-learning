'use client'

import { useState, useEffect, useRef } from 'react'
import { MantineProvider, Button, Checkbox, Paper, Flex, Grid, Text, Image, Container, Loader, Input } from '@mantine/core'
import theme from './Theme.tsx'
import Client from './Client.tsx'
import UrlInput from './UrlInput.tsx'
import Markdown from './Markdown.tsx'
import AudioControl from './Audio.tsx'

export default function DocumentViewer() {
  const [requestId, setRequestId] = useState('')
  const [pageNum, setPageNum] = useState(1)
  const [maxPageNum, setMaxPageNum] = useState(1)
  const [explanation, setExplanation] = useState('')
  const [imageUrl, setImageUrl] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [isInitialized, setIsInitialized] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [volume, setVolume] = useState(0.5)
  const [speaking, setSpeaking] = useState(true)
  const [autoAdvance, setAutoAdvance] = useState(false)
  const [gotoPage, setGotoPage] = useState(1)
  const containerRef = useRef<HTMLDivElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const client = new Client()

  const generate = async (reqId: string, page: number, regenerate: boolean) => {
    var error: boolean = false
    setIsLoading(true)
    const runExplain = (regenerate ? client.regenerate(reqId, page) : client.explain(reqId, page)).then((explanation: string | null) => {
      if (explanation !== null) {
        setExplanation(explanation)
      } else {
        error = true
      }
    })

    const runImage = client.image(reqId, page).then((imageUrl: string | null) => {
      if (imageUrl !== null) {
        setImageUrl(imageUrl)
      } else {
        error = true
      }
    })
    await Promise.all([runExplain, runImage])
    await client.audio(reqId, page).then((audioUrl: string | null) => {
      if (audioUrl !== null) {
        setAudioUrl(audioUrl)
      } else {
        error = true
      }
    })

    setIsLoading(false)
    if (error) {
      console.error('Error fetching explanation and image')
    }
  }

  const fetchExplanationAndImage = async (reqId: string, page: number) => {
    return generate(reqId, page, false)
  }

  const isFirstPage = () => pageNum === 1
  const isLastPage = () => pageNum === maxPageNum
  const isValidPage = (page: number) => page >= 1 && page <= maxPageNum
  const regenerate = async (reqId: string, page: number) => {
    generate(reqId, page, true)
  }

  const changePage = (newPage: number) => {
    if (isValidPage(newPage)) {
      setPageNum(newPage)
      setGotoPage(newPage)
      fetchExplanationAndImage(requestId, newPage)
    }
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'PageUp' || event.key === 'ArrowUp') {
        changePage(pageNum - 1)
      } else if (event.key === 'PageDown' || event.key === 'ArrowDown') {
        changePage(pageNum + 1)
      }
    }

    if (isInitialized) {
      window.addEventListener('keydown', handleKeyDown)
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [pageNum, isInitialized])

  useEffect(() => {
    const handleAudioEnded = () => {
      if (autoAdvance) {
        changePage(pageNum + 1)
      }
    }

    const audioElement = audioRef.current
    if (audioElement) {
      audioElement.addEventListener('ended', handleAudioEnded)
    }

    return () => {
      if (audioElement) {
        audioElement.removeEventListener('ended', handleAudioEnded)
      }
    }
  }, [pageNum, requestId, autoAdvance, audioRef])

  useEffect(() => {
    if (audioRef.current) {
      if (speaking) {
        audioRef.current.play()
      } else {
        audioRef.current.pause()
      }
    }
  }, [speaking])

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume
    }
  }, [volume])

  useEffect(() => {
    const audioElement = audioRef.current;
    if (audioElement) {
      const handlePlay = () => setSpeaking(true);
      const handlePause = () => {
        if (!(audioElement.ended && autoAdvance && !isLastPage())) {
          setSpeaking(false);
        }
      }

      audioElement.addEventListener('play', handlePlay);
      audioElement.addEventListener('pause', handlePause);

      return () => {
        audioElement.removeEventListener('play', handlePlay);
        audioElement.removeEventListener('pause', handlePause);
      };
    }
  }, [audioRef]);

  return (
    <MantineProvider theme={theme}>
      <Container ref={containerRef}>
        <Grid>
          <Grid.Col span={6}>
            <UrlInput onGetResult={({ request_id, page_num }) => {
              setRequestId(request_id)
              setMaxPageNum(page_num)
              setPageNum(1)
              setIsInitialized(true)
              fetchExplanationAndImage(request_id, 1)
            }} />

            <Flex gap="xs">
              <Button onClick={() => changePage(pageNum - 1)} disabled={isFirstPage()}>前ページ</Button>
              <Button onClick={() => changePage(pageNum + 1)} disabled={isLastPage()}>次ページ</Button>
              <Button onClick={() => changePage(Math.floor(Math.random() * maxPageNum) + 1)}>ランダム</Button>
              <Checkbox
                label="常に次のページへ"
                checked={autoAdvance}
                onChange={(event) => setAutoAdvance(event.currentTarget.checked)}
              />
            </Flex>

            {isInitialized && <>
              <Text>Page: {pageNum} / {maxPageNum} </Text>
              <Flex gap="xs">
                <Input type="number" value={gotoPage} onChange={(event) => setGotoPage(parseInt(event.currentTarget.value))} />
                <Button onClick={() => changePage(gotoPage)} disabled={!isValidPage(gotoPage)}>指定したページへ</Button>
              </Flex></>}
          </Grid.Col>

          <Grid.Col span={6}>
            <audio ref={audioRef} src={audioUrl} autoPlay></audio>
            <AudioControl
              volume={volume}
              setVolume={setVolume}
              speaking={speaking}
              setSpeaking={setSpeaking}
            />

            <Flex gap="xs">
              <Button onClick={() => regenerate(requestId, pageNum)}>再生成</Button>
            </Flex>
          </Grid.Col>
        </Grid>
        <Grid type="container">
          <Grid.Col span={6}>
            <Paper>
              {isInitialized && (
                <>
                  {isLoading ? <Loader /> : <Markdown> {explanation} </Markdown>}
                </>
              )}
            </Paper>
          </Grid.Col>
          <Grid.Col span={6}>
            {isInitialized && imageUrl && (
              <Image src={imageUrl} alt={`Page ${pageNum}`} fit="contain" />
            )}
          </Grid.Col>
        </Grid>
      </Container>
    </MantineProvider >
  )
}
