# Data Model: AI-Native Technical Textbook Platform - Phase 1

## Entities

### Chapter
- **Fields**:
  - id: string (unique identifier for the chapter)
  - title: string (chapter title)
  - part: string (which part of the textbook this chapter belongs to)
  - content: string (markdown content of the chapter)
  - learningObjectives: array of strings (learning objectives for the chapter)
  - codeExamples: array of code blocks (Python/ROS examples)
  - exercises: array of exercise objects (exercises for the chapter)
  - summary: string (chapter summary)
  - prerequisites: array of strings (prerequisites for this chapter)
  - nextChapter: string (reference to the next chapter in sequence)

- **Validation rules**:
  - id must be unique across all chapters
  - title, content, and learningObjectives are required
  - part must be one of the 4 defined parts (Foundations, ROS 2, Digital Twins, Vision-Language-Action)
  - content must be valid markdown

### Part
- **Fields**:
  - id: string (unique identifier for the part)
  - title: string (part title)
  - chapters: array of chapter references (ordered list of chapters in this part)
  - description: string (description of the part)

- **Validation rules**:
  - id must be unique across all parts
  - title and description are required
  - chapters array must contain valid chapter references
  - There must be exactly 4 parts

### Exercise
- **Fields**:
  - id: string (unique identifier for the exercise)
  - title: string (exercise title)
  - description: string (detailed description of the exercise)
  - difficulty: string (difficulty level: beginner, intermediate, advanced)
  - type: string (type of exercise: practical, theoretical, code-based)
  - solution: string (solution or answer to the exercise)
  - chapterId: string (reference to the chapter this exercise belongs to)

- **Validation rules**:
  - id must be unique across all exercises
  - title, description, and chapterId are required
  - difficulty must be one of: beginner, intermediate, advanced
  - type must be one of: practical, theoretical, code-based

### User
- **Fields**:
  - id: string (unique identifier for the user)
  - role: string (user role: student, practitioner, educator)
  - progress: array of progress objects (tracking user progress through chapters)

- **Validation rules**:
  - id must be unique across all users
  - role must be one of: student, practitioner, educator
  - For Phase 1, user tracking is read-only (no authentication required)

### Progress
- **Fields**:
  - userId: string (reference to the user)
  - chapterId: string (reference to the chapter)
  - completed: boolean (whether the chapter has been completed)
  - lastAccessed: date (when the user last accessed this chapter)

- **Validation rules**:
  - userId and chapterId combination must be unique
  - completed must be a boolean value

## Relationships

- **Part** 1-to-many **Chapter**: Each part contains multiple chapters in a specific order
- **Chapter** 1-to-many **Exercise**: Each chapter contains multiple exercises
- **User** 1-to-many **Progress**: Each user has progress tracking for multiple chapters
- **Chapter** 1-to-many **Progress**: Each chapter has progress tracking for multiple users

## State Transitions

### Chapter State
- **Draft**: Content is being created/edited
- **Review**: Content is under review for technical accuracy
- **Published**: Content is available to users
- **Archived**: Content is no longer actively maintained

### Progress State
- **Not Started**: User has not accessed the chapter
- **In Progress**: User has started reading the chapter
- **Completed**: User has finished the chapter

## Constraints

1. Each chapter must belong to exactly one part
2. Each exercise must belong to exactly one chapter
3. Content must be structured in the 4-part, 16-chapter format specified
4. All code examples must be in Python or ROS (as specified in requirements)
5. Exercises must align with the learning objectives of their respective chapters