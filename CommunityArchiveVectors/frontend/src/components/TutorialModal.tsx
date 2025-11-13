'use client'

import { useState, useEffect } from 'react'
import { X, ChevronLeft, ChevronRight, Info } from 'lucide-react'
import { lugrasimo } from '../app/fonts'

interface TutorialStep {
  title: string
  content: string
  highlight?: string
  image?: string
}

const tutorialSteps: TutorialStep[] = [
  {
    title: "Welcome to the Constellation of People",
    content: "This visualization shows 5.7 million tweets organized into communities and topics across 2012-2025. Explore how online conversations cluster and evolve over time.",
  },
  {
    title: "Selecting a Year",
    content: "Use the year buttons at the bottom to explore different time periods. Each year contains tweets organized into communities based on their semantic similarity.",
    highlight: "year-selector"
  },
  {
    title: "Communities & Topics",
    content: "Communities are groups of users clustered by the Louvain algorithm based on how much they interact with each other. Within each community, tweets are further clustered into specific topics using K-means clustering.",
  },
  {
    title: "How It Works: Embeddings",
    content: "Each tweet is converted into a 1024-dimensional vector (embedding) that captures its semantic meaning. Similar tweets have similar embeddings, allowing us to group related content together.",
  },
  {
    title: "How It Works: Community Detection",
    content: "The Louvain algorithm identifies communities by maximizing modularity - finding groups of users who interact more frequently with each other than with users outside their group. Users are connected based on replies, mentions, and retweets. Learn more: <a href='https://en.wikipedia.org/wiki/Louvain_method' target='_blank' class='text-amber-400 hover:underline'>Louvain Method</a>",
  },
  {
    title: "How It Works: Topic Clustering",
    content: "Within each community, K-means clustering groups tweets into specific topics based on their embedding similarity. K-means finds k cluster centers that minimize the distance to their assigned tweets. Learn more: <a href='https://en.wikipedia.org/wiki/K-means_clustering' target='_blank' class='text-amber-400 hover:underline'>K-means Clustering</a>",
  },
  {
    title: "Viewing Topics",
    content: "Click on any community card to see its topics. Each topic shows the number of tweets and a Claude-generated description of the conversation theme.",
  },
  {
    title: "Reading Tweets",
    content: "Click on a topic to view the actual tweets. You'll see up to 50 tweets from that cluster with full text, usernames, and timestamps.",
  },
  {
    title: "Additional Resources",
    content: `
      <div class="space-y-2">
        <p class="font-semibold mb-2">Learn More:</p>
        <ul class="list-disc list-inside space-y-1 text-sm">
          <li><a href="https://en.wikipedia.org/wiki/Word_embedding" target="_blank" class="text-amber-400 hover:underline">Word Embeddings (Wikipedia)</a></li>
          <li><a href="https://arxiv.org/abs/1301.3781" target="_blank" class="text-amber-400 hover:underline">Word2Vec Paper (Mikolov et al.)</a></li>
          <li><a href="https://arxiv.org/abs/0803.0476" target="_blank" class="text-amber-400 hover:underline">Fast Unfolding Communities (Louvain, 2008)</a></li>
          <li><a href="https://scikit-learn.org/stable/modules/clustering.html#k-means" target="_blank" class="text-amber-400 hover:underline">K-means Clustering (scikit-learn)</a></li>
          <li><a href="https://umap-learn.readthedocs.io/" target="_blank" class="text-amber-400 hover:underline">UMAP Dimensionality Reduction</a></li>
        </ul>
      </div>
    `,
  }
]

interface TutorialModalProps {
  isOpen?: boolean
  onClose?: () => void
  onOpen?: () => void
  showButton?: boolean
}

export default function TutorialModal({ isOpen: externalIsOpen, onClose, onOpen, showButton = false }: TutorialModalProps = {}) {
  const [internalIsOpen, setInternalIsOpen] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)

  // Use external control if provided, otherwise use internal state
  const isOpen = externalIsOpen !== undefined ? externalIsOpen : internalIsOpen

  const nextStep = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const closeTutorial = () => {
    if (onClose) {
      onClose()
    } else {
      setInternalIsOpen(false)
    }
    setCurrentStep(0)
  }

  const openTutorial = () => {
    if (onOpen) {
      onOpen()
    } else {
      setInternalIsOpen(true)
    }
    setCurrentStep(0)
  }

  const currentStepData = tutorialSteps[currentStep]
  const progress = ((currentStep + 1) / tutorialSteps.length) * 100

  return (
    <>
      {/* Info Button - Retro Style */}
      {showButton && (
        <button
          onClick={openTutorial}
          className={`${lugrasimo.className} fixed top-4 right-4 z-50 text-[#d4a574] rounded-none px-4 py-3 shadow-[0_0_15px_rgba(212,165,116,0.5)] transition-all duration-200 hover:scale-105 flex items-center gap-2 group border-2 border-[#6b9080] bg-black/80 hover:bg-[#6b9080]/20`}
          style={{
            textShadow: '0 0 10px rgba(212,165,116,0.8)',
            boxShadow: '0 0 20px rgba(107,144,128,0.3), inset 0 0 20px rgba(107,144,128,0.1)'
          }}
          title="How it works"
        >
          <Info className="w-5 h-5" />
          <span className="max-w-0 overflow-hidden group-hover:max-w-xs transition-all duration-300 whitespace-nowrap text-sm">
            HOW IT WORKS
          </span>
        </button>
      )}

      {/* Tutorial Modal - Retro CRT Style */}
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fadeIn">
          <div
            className="bg-black rounded-none shadow-2xl max-w-2xl w-full border-2 border-[#6b9080] animate-slideUp relative"
            style={{
              boxShadow: '0 0 40px rgba(107,144,128,0.4), inset 0 0 40px rgba(107,144,128,0.05)',
            }}
          >
            {/* Scanline effect overlay */}
            <div
              className="absolute inset-0 pointer-events-none opacity-10"
              style={{
                backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(107,144,128,0.1) 2px, rgba(107,144,128,0.1) 4px)'
              }}
            />

            {/* Header */}
            <div className={`${lugrasimo.className} flex items-center justify-between p-6 border-b-2 border-[#6b9080]/50`}>
              <div className="flex items-center gap-3">
                <div
                  className="border-2 border-[#6b9080] rounded-none p-2 bg-[#6b9080]/20"
                  style={{
                    boxShadow: '0 0 15px rgba(107,144,128,0.3)'
                  }}
                >
                  <Info className="w-5 h-5 text-[#d4a574]" />
                </div>
                <div>
                  <h2
                    className="text-xl font-bold text-[#d4a574]"
                    style={{
                      textShadow: '0 0 10px rgba(212,165,116,0.8)'
                    }}
                  >
                    {currentStepData.title}
                  </h2>
                  <p className="text-sm text-[#6b89a8]">
                    STEP {currentStep + 1} OF {tutorialSteps.length}
                  </p>
                </div>
              </div>
              <button
                onClick={closeTutorial}
                className="text-[#6b89a8] hover:text-[#d4a574] transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Progress Bar - Retro */}
            <div className="w-full h-2 bg-black/50 border-y border-[#6b9080]/50">
              <div
                className="h-full bg-[#d4a574] transition-all duration-300 ease-out relative"
                style={{
                  width: `${progress}%`,
                  boxShadow: '0 0 10px rgba(212,165,116,0.6)'
                }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#d4a574]/30 to-transparent animate-shimmer" />
              </div>
            </div>

            {/* Content */}
            <div className="p-8 min-h-[300px] relative">
              <div
                className="text-[#e8dcc8] leading-relaxed animate-fadeIn"
                style={{
                  textShadow: '0 0 5px rgba(232,220,200,0.2)',
                  fontFamily: 'monospace',
                  fontSize: '15px'
                }}
                dangerouslySetInnerHTML={{ __html: currentStepData.content }}
              />

              {currentStepData.highlight && (
                <div
                  className="mt-4 p-3 bg-[#6b9080]/20 border border-[#6b9080]/50 rounded-none text-sm text-[#d4a574]"
                  style={{
                    boxShadow: '0 0 10px rgba(107,144,128,0.2), inset 0 0 10px rgba(107,144,128,0.1)'
                  }}
                >
                  💡 LOOK FOR: <span className="font-semibold">{currentStepData.highlight}</span>
                </div>
              )}
            </div>

            {/* Footer Navigation */}
            <div className={`${lugrasimo.className} flex items-center justify-between p-6 border-t-2 border-[#6b9080]/50`}>
              <button
                onClick={prevStep}
                disabled={currentStep === 0}
                className={`flex items-center gap-2 px-4 py-2 border-2 rounded-none transition-all ${
                  currentStep === 0
                    ? 'text-gray-600 border-gray-600 cursor-not-allowed'
                    : 'text-[#d4a574] border-[#6b9080] hover:bg-[#6b9080]/20 hover:shadow-[0_0_15px_rgba(107,144,128,0.3)]'
                }`}
                style={currentStep > 0 ? {
                  textShadow: '0 0 8px rgba(212,165,116,0.6)'
                } : {}}
              >
                <ChevronLeft className="w-4 h-4" />
                PREVIOUS
              </button>

              <div className="flex gap-2">
                {tutorialSteps.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentStep(index)}
                    className={`w-2 h-2 rounded-none border transition-all ${
                      index === currentStep
                        ? 'bg-[#d4a574] border-[#d4a574] w-8 shadow-[0_0_10px_rgba(212,165,116,0.6)]'
                        : 'bg-[#6b9080]/30 border-[#6b9080]/50 hover:bg-[#6b9080]/50'
                    }`}
                    aria-label={`Go to step ${index + 1}`}
                  />
                ))}
              </div>

              {currentStep < tutorialSteps.length - 1 ? (
                <button
                  onClick={nextStep}
                  className="flex items-center gap-2 px-4 py-2 border-2 border-[#6b9080] text-[#d4a574] rounded-none transition-all hover:bg-[#6b9080]/20"
                  style={{
                    textShadow: '0 0 8px rgba(212,165,116,0.6)',
                    boxShadow: '0 0 15px rgba(107,144,128,0.3)'
                  }}
                >
                  NEXT
                  <ChevronRight className="w-4 h-4" />
                </button>
              ) : (
                <button
                  onClick={closeTutorial}
                  className="px-6 py-2 border-2 border-[#6b9080] text-[#d4a574] rounded-none transition-all font-semibold hover:bg-[#6b9080]/20"
                  style={{
                    textShadow: '0 0 8px rgba(212,165,116,0.6)',
                    boxShadow: '0 0 15px rgba(107,144,128,0.3)'
                  }}
                >
                  ENGAGE!
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }

        .animate-slideUp {
          animation: slideUp 0.4s ease-out;
        }

        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </>
  )
}
